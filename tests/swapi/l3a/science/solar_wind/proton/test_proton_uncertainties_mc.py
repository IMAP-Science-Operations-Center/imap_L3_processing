"""Monte Carlo calibration test for `derive_uncertainties` (Huber-White sandwich).

Mirrors the validation in `docs/swapi/figure_src/plot_uncertainty_mc.py`: hold one
typical solar-wind ground truth fixed, sample Poisson noise across many synthetic
5-sweep chunks, fit each one, and check that the *median* per-fit sandwich-
estimated sigma is close to the *empirical* sigma of the fitted-parameter
distribution for every fit parameter (n, T, v_R, v_T, v_N).

If the sandwich estimator is well-calibrated under the figure's noise model, the
median estimated/empirical ratio is within a few percent of unity (per the doc).
This test uses a much smaller MC sample and a wider tolerance so it can run as
a unit test, but is still tight enough to catch the multi-x mis-scaling we
observe on real CDFs.
"""

import multiprocessing
import os
import types
import unittest
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    SWAPI_LIVETIME_S,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    optimize_solar_wind_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.basin_hopping import (
    escape_local_minimum,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.uncertainties import (
    derive_uncertainties,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from tests.swapi._swapi_test_helpers import swapi_response


_N_SWEEPS = 5
_SWEEP_DURATION_S = 12.0
_BINS_PER_SWEEP = 72
_DT_S = _SWEEP_DURATION_S / _BINS_PER_SWEEP
_SPIN_PERIOD_S = 15.13

_BIN_INDICES_IN_SWEEP = np.arange(1, 63)
_VOLTAGES = np.array(
    [
        9895.52,
        9088.69,
        8348.80,
        7667.55,
        7042.16,
        6469.31,
        5941.77,
        5457.31,
        5013.22,
        4603.65,
        4230.77,
        3886.92,
        3569.16,
        3278.72,
        3011.13,
        2766.25,
        2539.54,
        2333.83,
        2144.24,
        1969.31,
        1808.74,
        1660.86,
        1525.75,
        1401.82,
        1287.58,
        1182.24,
        1085.15,
        995.55,
        914.31,
        839.94,
        771.70,
        709.46,
        651.59,
        598.47,
        549.91,
        505.12,
        463.89,
        425.92,
        391.18,
        359.35,
        329.94,
        303.02,
        278.25,
        255.55,
        234.77,
        215.61,
        197.95,
        181.82,
        167.04,
        153.46,
        140.91,
        129.50,
        118.91,
        109.20,
        100.30,
        92.11,
        84.61,
        77.73,
        71.40,
        65.59,
        60.23,
        55.34,
    ]
)

# RTN -> SWAPI rotation at the chunk anchor time, copied verbatim from
# plot_uncertainty_mc.py so the synthetic fits exercise the same geometry.
_ANCHOR_ROTATION_MATRIX = np.array(
    [
        [+0.0705, +0.9157, +0.3955],
        [-0.9968, +0.0792, -0.0057],
        [-0.0365, -0.3939, +0.9184],
    ]
).T
_ANCHOR_TIME_S = 0.5 * _SWEEP_DURATION_S
_SPIN_OMEGA_RAD_S = -2.0 * np.pi / _SPIN_PERIOD_S

_TRUTH_DENSITY_CM3 = 5.0
_TRUTH_TEMPERATURE_K = 1.0e5
_TRUTH_BULK_RTN_KM_S = np.array([450.0, 5.0, -3.0])

# Noise model: Poisson on counts + 1% multiplicative log-normal jitter
# + a 10 Hz energy-independent Poisson floor. Same model used by the
# `plot_uncertainty_mc.py` validation figure in `docs/swapi/`.
_LOGNORMAL_REL_SIGMA = 0.01
_NOISE_FLOOR_HZ = 10.0

# N_MC=1000 gets the std-of-σ̂-estimator down to ~2.2% per parameter
# (= 1/√(2·(N-1))), so a ±10% calibration window is well above the floor.
# Run takes ~5 s on 18 cores (warm numba caches).
_N_MC_SAMPLES = 1000
# Normalized residual (fit − truth)/σ̂ should be N(0, 1) for a calibrated
# estimator. We require std ∈ [0.9, 1.1].
_NORM_RES_STD_TOLERANCE = 0.10


_worker_state: types.SimpleNamespace | None = None


def _compute_per_bin_rotation_matrices(n_sweeps, bin_indices_in_sweep):
    sweep_index = np.repeat(np.arange(n_sweeps), len(bin_indices_in_sweep))
    bin_index = np.tile(bin_indices_in_sweep, n_sweeps)
    sample_times_s = sweep_index * _SWEEP_DURATION_S + bin_index * _DT_S

    spin_axis = _ANCHOR_ROTATION_MATRIX[:, 1] / np.linalg.norm(
        _ANCHOR_ROTATION_MATRIX[:, 1]
    )
    delta_phi = _SPIN_OMEGA_RAD_S * (sample_times_s - _ANCHOR_TIME_S)

    ax, ay, az = spin_axis
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    sin_dp = np.sin(delta_phi)[:, None, None]
    one_minus_cos = (1.0 - np.cos(delta_phi))[:, None, None]
    rot = np.eye(3) + sin_dp * K + one_minus_cos * (K @ K)
    return rot @ _ANCHOR_ROTATION_MATRIX


def _initialize_worker_state():
    sr = swapi_response()
    all_esa_voltages = np.tile(_VOLTAGES, _N_SWEEPS)
    sr.warm_cache(all_esa_voltages)
    per_bin_rot = _compute_per_bin_rotation_matrices(_N_SWEEPS, _BIN_INDICES_IN_SWEEP)
    base_ctx = build_solar_wind_fit_context(
        count_rate=np.ones_like(all_esa_voltages),
        esa_voltage=all_esa_voltages,
        swapi_response=sr,
        central_effective_area_scale=1.0,
        rotation_matrices=per_bin_rot,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    truth_params = SolarWindParams(
        density=_TRUTH_DENSITY_CM3,
        bulk_velocity_rtn=_TRUTH_BULK_RTN_KM_S.copy(),
        temperature=_TRUTH_TEMPERATURE_K,
        mass=PROTON_MASS_KG,
    )
    ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(truth_params, base_ctx)
    truth_count_rates = ideal_rates * deadtime_factor(ideal_rates)

    global _worker_state
    _worker_state = types.SimpleNamespace(
        swapi_response=sr,
        all_esa_voltages=all_esa_voltages,
        per_bin_rotation_matrices=per_bin_rot,
        truth_count_rates=truth_count_rates,
    )


def _run_one(seed):
    ws = _worker_state
    rng = np.random.default_rng(seed)
    # Poisson on the signal counts.
    counts = rng.poisson(np.maximum(ws.truth_count_rates * SWAPI_LIVETIME_S, 0.0))
    count_rates = counts.astype(float) / SWAPI_LIVETIME_S
    # Multiplicative log-normal jitter (mean-1 corrected via -σ²/2 shift in log space).
    sigma_log = np.sqrt(np.log1p(_LOGNORMAL_REL_SIGMA**2))
    log_factor = rng.normal(-0.5 * sigma_log**2, sigma_log, count_rates.shape)
    count_rates = np.maximum(count_rates * np.exp(log_factor), 0.0)
    # Energy-independent Poisson floor — adds Poisson(floor·τ) counts at every bin.
    floor_counts = rng.poisson(
        _NOISE_FLOOR_HZ * SWAPI_LIVETIME_S, size=count_rates.shape
    )
    count_rates = count_rates + floor_counts.astype(float) / SWAPI_LIVETIME_S

    fit_ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=ws.all_esa_voltages,
        swapi_response=ws.swapi_response,
        central_effective_area_scale=1.0,
        rotation_matrices=ws.per_bin_rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    try:
        initial_guess = calculate_initial_guess(fit_ctx)
        first = optimize_solar_wind_params(initial_guess, fit_ctx)
        final = escape_local_minimum(first, fit_ctx)
    except Exception:
        return None
    if not final.success:
        return None

    n_sigma, T_sigma, v_cov = derive_uncertainties(final, fit_ctx)
    if not (
        np.isfinite(n_sigma) and np.isfinite(T_sigma) and np.all(np.isfinite(v_cov))
    ):
        return None
    v_sigma = np.sqrt(np.maximum(np.diag(v_cov), 0.0))
    sw = final.sw_params
    return (
        sw.density,
        sw.temperature,
        float(sw.bulk_velocity_rtn[0]),
        float(sw.bulk_velocity_rtn[1]),
        float(sw.bulk_velocity_rtn[2]),
        n_sigma,
        T_sigma,
        float(v_sigma[0]),
        float(v_sigma[1]),
        float(v_sigma[2]),
    )


class TestSandwichSigmaCalibratedAgainstNoisyMC(unittest.TestCase):
    """For a fixed solar-wind truth, the per-fit sandwich sigma should be
    calibrated against the actual MC scatter of fitted parameters.

    Calibration is checked via the std of normalized residuals
    (fit − truth)/σ̂ across MC realizations: a well-calibrated estimator
    produces N(0, 1) per parameter, i.e. std ≈ 1.0. We require std to be
    within ±10% of unity for every fitted parameter.

    Noise model matches `docs/swapi/figure_src/plot_uncertainty_mc.py`:
    Poisson on signal counts + 1% multiplicative log-normal jitter
    + 10 Hz energy-independent Poisson floor at every bin.
    """

    @classmethod
    def setUpClass(cls):
        if multiprocessing.get_start_method(allow_none=True) != "fork":
            multiprocessing.set_start_method("fork", force=True)
        _initialize_worker_state()
        ctx = multiprocessing.get_context("fork")
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            results = list(pool.map(_run_one, range(_N_MC_SAMPLES), chunksize=4))

        rows = [r for r in results if r is not None]
        if len(rows) < _N_MC_SAMPLES // 2:
            raise AssertionError(
                f"Too many fits failed: {_N_MC_SAMPLES - len(rows)}/{_N_MC_SAMPLES}; "
                "MC validation cannot proceed."
            )
        arr = np.array(rows)
        cls.fits = arr[:, :5]  # n, T, vR, vT, vN
        cls.sigmas = arr[:, 5:]
        cls.truth = np.array(
            [
                _TRUTH_DENSITY_CM3,
                _TRUTH_TEMPERATURE_K,
                _TRUTH_BULK_RTN_KM_S[0],
                _TRUTH_BULK_RTN_KM_S[1],
                _TRUTH_BULK_RTN_KM_S[2],
            ]
        )
        cls.normalized_residuals = (cls.fits - cls.truth) / cls.sigmas
        cls.norm_resid_std = np.std(cls.normalized_residuals, axis=0, ddof=1)
        cls.param_names = ["density", "temperature", "v_R", "v_T", "v_N"]

    def test_normalized_residual_std_close_to_unity(self):
        for name, std in zip(self.param_names, self.norm_resid_std):
            with self.subTest(param=name):
                self.assertAlmostEqual(
                    std,
                    1.0,
                    delta=_NORM_RES_STD_TOLERANCE,
                    msg=(
                        f"std of normalized residual (fit − truth)/σ̂ for {name} "
                        f"is {std:.3g}; expected within "
                        f"{_NORM_RES_STD_TOLERANCE:.0%} of 1.0. "
                        f"Bars are off by a factor of {std:.3g}."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
