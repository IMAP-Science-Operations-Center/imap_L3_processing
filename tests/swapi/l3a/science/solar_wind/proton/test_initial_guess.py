"""Tests for `solar_wind.proton.initial_guess` — `calculate_initial_guess` and the
`_gaussian_refine_bulk_speed_and_temperature` helper.

The doc contract (`docs/swapi/solar-wind-moments.md` §Initial Guess) is:
1. Bulk-speed seed = proton speed at the ESA voltage of the highest count-rate bin.
2. Temperature seed = max(1 eV, 60_000 K · (v_b / 400 km/s)²).
3. Refine `(v_b, T)` via a Gaussian curve fit; on failure, keep the seeds.
4. Velocity direction is anti-parallel to the chunk-mean spin axis (body +Y in RTN).
5. Density is the optimal scale of the unit-density forward-model rates against
   the observed count rates.

Tests use the real shipped instrument-team CSVs to build a `SwapiResponse` and
the real `build_solar_wind_fit_context` factory. The Gaussian refiner is patched
in the seed-construction tests so the seeds passed to it are observable directly;
elsewhere the refiner runs unmocked. Helper-level tests
(`_gaussian_refine_*`) are exercised directly on synthetic 1-D arrays."""

import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.initial_guess import (
    INITIAL_TEMPERATURE_FLOOR_K,
    _gaussian_refine_bulk_speed_and_temperature,
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    SolarWindParams,
    temperature_to_thermal_speed,
)
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.response.speed_calculation import SWAPI_K_FACTOR
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from tests.test_helpers import get_test_instrument_team_data_path


# RTN → SWAPI rotation. Body +Y (the SWAPI boresight / spin axis) in RTN is
# column 1 of the transpose, i.e. -R̂_RTN. The solar wind direction (anti-
# parallel to the spin axis) is therefore +R̂.
_R_BASE_RTN_TO_SWAPI = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)


def _load_swapi_response() -> SwapiResponse:
    """Build a real `SwapiResponse` from the shipped instrument-team CSVs."""
    return SwapiResponse.from_files(
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
        ),
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
        ),
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
        ),
    )


def _esa_voltage_for_proton_speed(speed_km_s: float) -> float:
    """Inverse of `esa_voltage_to_proton_speed` — pick the ESA voltage whose
    central proton speed is exactly `speed_km_s`. Used to place the count-rate
    peak at a known voltage."""
    return float(
        PROTON_MASS_KG
        * (speed_km_s * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _spin_rotation_matrices(n: int) -> np.ndarray:
    """SWAPI→RTN rotation matrices for `n` consecutive bins. Body +Y (the
    SWAPI boresight / spin axis) lies along -R̂_RTN on every bin, so the
    chunk-mean spin axis is exactly -R̂."""
    return np.tile(_R_BASE_RTN_TO_SWAPI.T, (n, 1, 1))


def _build_proton_ctx(count_rate: np.ndarray, esa_voltage: np.ndarray):
    """Build a real proton-species `SolarWindFitContext` from the shipped CSVs.

    `count_rate`, `esa_voltage`, and the rotation matrices are all (N,)/(N,3,3)
    aligned in the standard L2 layout — one entry per bin, multiple sweeps
    flattened into a single axis."""
    response = _load_swapi_response()
    response.warm_cache(esa_voltage)
    ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=response,
        central_effective_area_scale=1.0,
        rotation_matrices=_spin_rotation_matrices(len(esa_voltage)),
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    return ctx


def _make_synthetic_ctx_at_known_truth(truth: SolarWindParams,
                                       n_bins: int = 71):
    """Build a context whose `count_rate` array is the noiseless ideal forward
    model evaluated at `truth`."""
    bulk_speed = float(np.linalg.norm(truth.bulk_velocity_rtn))
    # Wide enough to bracket ±5σ at T=1e5 K, narrow enough to keep all bins
    # on-instrument.
    speed_grid = np.linspace(0.4 * bulk_speed, 1.6 * bulk_speed, n_bins)
    voltages = np.array(
        [_esa_voltage_for_proton_speed(s) for s in speed_grid]
    )

    # Build a placeholder ctx, evaluate the forward model at `truth` to get
    # ideal rates, then build the production ctx with those rates as
    # `count_rate`.
    placeholder = _build_proton_ctx(np.ones_like(voltages), voltages)
    ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(truth, placeholder)
    return _build_proton_ctx(ideal_rates, voltages)


class TestGaussianRefineRecoversKnownGaussian(unittest.TestCase):
    """`_gaussian_refine_bulk_speed_and_temperature(speed, count_rate, ...)`
    fits an arbitrarily normalized Gaussian and returns the fitted
    `(bulk_speed, T)`. On a noiseless Gaussian count-rate spectrum the fit
    must recover the true mean and σ to high precision."""

    def test_recovers_true_mean_and_temperature_on_noiseless_gaussian(self):
        true_bulk_speed = 450.0
        true_temperature_k = 1.5e5
        true_sigma = temperature_to_thermal_speed(PROTON_MASS_KG,
                                                  true_temperature_k)

        # 6σ → tail < 1e-8 of peak.
        speed = np.linspace(true_bulk_speed - 6 * true_sigma,
                            true_bulk_speed + 6 * true_sigma, 200)
        count_rate = 1000.0 * np.exp(
            -0.5 * ((speed - true_bulk_speed) / true_sigma) ** 2
        )

        # Seeds are deliberately offset; a working fit ignores them and
        # converges on the true values.
        bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
            speed,
            count_rate,
            bulk_speed_seed=380.0,
            temperature_seed=5e4,
            mass_kg=PROTON_MASS_KG,
        )

        np.testing.assert_allclose(bulk_speed_fit, true_bulk_speed, rtol=1e-7)
        np.testing.assert_allclose(temperature_fit, true_temperature_k, rtol=1e-7)


class TestGaussianRefineFallback(unittest.TestCase):
    """When `scipy.optimize.curve_fit` cannot produce a finite fit (e.g. there
    are too few valid bins, or all bins are NaN), the refiner returns the
    seed values unchanged — the doc states the original seeds are kept on
    failure."""

    def test_returns_seeds_when_fewer_than_4_valid_bins(self):
        # Fewer than 4 valid bins.
        speed = np.array([400.0, 450.0, 500.0])
        count_rate = np.array([1.0, 2.0, 1.0])
        bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
            speed,
            count_rate,
            bulk_speed_seed=440.0,
            temperature_seed=1.2e5,
            mass_kg=PROTON_MASS_KG,
        )
        self.assertEqual(bulk_speed_fit, 440.0)
        self.assertEqual(temperature_fit, 1.2e5)

    def test_returns_seeds_when_all_counts_are_nan(self):
        # All counts NaN — no valid bins.
        speed = np.linspace(300.0, 600.0, 10)
        count_rate = np.full_like(speed, np.nan)
        bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
            speed,
            count_rate,
            bulk_speed_seed=420.0,
            temperature_seed=8e4,
            mass_kg=PROTON_MASS_KG,
        )
        self.assertEqual(bulk_speed_fit, 420.0)
        self.assertEqual(temperature_fit, 8e4)

    def test_returns_seeds_when_count_rate_is_all_zero(self):
        # All counts zero — no peak to fit.
        speed = np.linspace(300.0, 600.0, 30)
        count_rate = np.zeros_like(speed)
        bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
            speed,
            count_rate,
            bulk_speed_seed=455.0,
            temperature_seed=9e4,
            mass_kg=PROTON_MASS_KG,
        )
        self.assertAlmostEqual(bulk_speed_fit, 455.0)
        self.assertAlmostEqual(temperature_fit, 9e4)

    def test_returns_seeds_when_curve_fit_raises_runtime_error(self):
        # `scipy.optimize.curve_fit` raises `RuntimeError` when the Levenberg-
        # Marquardt loop hits its max-iters cap without converging. The
        # refiner catches this and falls back to the seeds rather than
        # bubbling the exception up to `calculate_initial_guess`.
        speed = np.linspace(300.0, 600.0, 30)
        count_rate = np.linspace(0.1, 0.5, 30)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess.scipy.optimize.curve_fit",
            side_effect=RuntimeError("Optimal parameters not found"),
        ):
            bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
                speed,
                count_rate,
                bulk_speed_seed=460.0,
                temperature_seed=8e4,
                mass_kg=PROTON_MASS_KG,
            )
        self.assertEqual(bulk_speed_fit, 460.0)
        self.assertEqual(temperature_fit, 8e4)

    def test_returns_seeds_when_curve_fit_raises_value_error(self):
        # `curve_fit` can also raise `ValueError` for ill-conditioned inputs
        # (e.g. NaN in the seed Jacobian). Fallback path is the same as
        # `RuntimeError`.
        speed = np.linspace(300.0, 600.0, 30)
        count_rate = np.linspace(0.1, 0.5, 30)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess.scipy.optimize.curve_fit",
            side_effect=ValueError("ydata contains NaNs"),
        ):
            bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
                speed,
                count_rate,
                bulk_speed_seed=475.0,
                temperature_seed=1.1e5,
                mass_kg=PROTON_MASS_KG,
            )
        self.assertEqual(bulk_speed_fit, 475.0)
        self.assertEqual(temperature_fit, 1.1e5)

    def test_returns_seeds_when_curve_fit_returns_non_finite_speed(self):
        # `curve_fit` can converge to non-finite parameters (e.g. NaN mean)
        # without raising — the post-fit guard rejects this and falls back
        # to the seeds.
        speed = np.linspace(300.0, 600.0, 30)
        count_rate = np.linspace(0.1, 0.5, 30)
        # Return shape is `(popt, pcov)` — `popt` is a 3-vector and `pcov`
        # is unused on the fallback path.
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess.scipy.optimize.curve_fit",
            return_value=(np.array([1.0, np.nan, 100.0]), None),
        ):
            bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
                speed,
                count_rate,
                bulk_speed_seed=465.0,
                temperature_seed=9.5e4,
                mass_kg=PROTON_MASS_KG,
            )
        self.assertEqual(bulk_speed_fit, 465.0)
        self.assertEqual(temperature_fit, 9.5e4)

    def test_returns_seeds_when_curve_fit_returns_non_positive_sigma(self):
        # `sigma_fit` is taken in absolute value, so a negative σ at fit
        # convergence is treated as positive. But `σ == 0` (after
        # `abs(...)`) signals a degenerate fit and triggers the fallback
        # — sigma is the Maxwellian width, and zero width gives an
        # undefined temperature.
        speed = np.linspace(300.0, 600.0, 30)
        count_rate = np.linspace(0.1, 0.5, 30)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess.scipy.optimize.curve_fit",
            return_value=(np.array([1.0, 450.0, 0.0]), None),
        ):
            bulk_speed_fit, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
                speed,
                count_rate,
                bulk_speed_seed=440.0,
                temperature_seed=1.0e5,
                mass_kg=PROTON_MASS_KG,
            )
        self.assertEqual(bulk_speed_fit, 440.0)
        self.assertEqual(temperature_fit, 1.0e5)


class TestGaussianRefineTemperatureFloor(unittest.TestCase):
    """The doc states the refined temperature is clamped at 1 eV
    (`INITIAL_TEMPERATURE_FLOOR_K`). When the fitted σ implies T below the
    floor, the helper returns `INITIAL_TEMPERATURE_FLOOR_K`."""

    def test_floors_temperature_at_1_eV(self):
        # σ_v corresponding to a temperature below the floor.
        true_bulk_speed = 450.0
        true_sigma_below_floor = temperature_to_thermal_speed(
            PROTON_MASS_KG, 0.01 * INITIAL_TEMPERATURE_FLOOR_K
        )
        # 6σ → tail < 1e-8 of peak.
        speed = np.linspace(true_bulk_speed - 6 * true_sigma_below_floor,
                            true_bulk_speed + 6 * true_sigma_below_floor, 200)
        count_rate = 1000.0 * np.exp(
            -0.5 * ((speed - true_bulk_speed) / true_sigma_below_floor) ** 2
        )

        _, temperature_fit = _gaussian_refine_bulk_speed_and_temperature(
            speed,
            count_rate,
            bulk_speed_seed=true_bulk_speed,
            temperature_seed=1e5,
            mass_kg=PROTON_MASS_KG,
        )
        self.assertEqual(temperature_fit, INITIAL_TEMPERATURE_FLOOR_K)


class TestCalculateInitialGuessSeeds(unittest.TestCase):
    """Pin the doc-specified seed construction in `calculate_initial_guess`:
    bulk-speed seed at the peak ESA voltage, temperature seed
    `60_000 · (v/400)²` floored at 1 eV.

    These tests patch `_gaussian_refine_bulk_speed_and_temperature` so the
    seeds passed to it are observable directly — independent of how the
    Gaussian fit happens to converge for any given input."""

    def _ctx_with_peak_at(self, peak_speed_kms: float):
        """Build a context whose count-rate spectrum has its maximum at a
        bin whose ESA voltage corresponds to `peak_speed_kms`. The shape of
        the spectrum doesn't matter — we patch out the refiner."""
        speed_grid = np.linspace(0.4 * peak_speed_kms, 1.6 * peak_speed_kms,
                                 31)
        voltages = np.array(
            [_esa_voltage_for_proton_speed(s) for s in speed_grid]
        )
        peak_idx = len(speed_grid) // 2  # middle bin
        voltages[peak_idx] = _esa_voltage_for_proton_speed(peak_speed_kms)
        count_rate = np.linspace(0.1, 0.5, len(voltages))
        count_rate[peak_idx] = 100.0  # Unambiguous global maximum.
        return _build_proton_ctx(count_rate, voltages)

    def test_bulk_speed_seed_is_proton_speed_at_peak_voltage(self):
        peak_speed = 480.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, 1e5),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        # Args: (speed, count_rate, bulk_speed_seed, temperature_seed, mass_kg)
        args = patched_refine.call_args.args
        bulk_speed_seed_arg = args[2]
        np.testing.assert_allclose(bulk_speed_seed_arg, peak_speed, rtol=1e-12)

    def test_temperature_seed_uses_documented_speed_squared_formula(self):
        # T_seed = 60_000 · (v / 400)² when above the 1 eV floor.
        peak_speed = 480.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, 1e5),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        temperature_seed_arg = patched_refine.call_args.args[3]
        expected_temperature = 60_000.0 * (peak_speed / 400.0) ** 2
        np.testing.assert_allclose(temperature_seed_arg, expected_temperature,
                                   rtol=1e-12)

    def test_temperature_seed_floors_at_one_ev(self):
        # At v_b small enough that 60_000 · (v/400)² < 1 eV, the seed is
        # clamped to INITIAL_TEMPERATURE_FLOOR_K.
        peak_speed = 100.0
        ctx = self._ctx_with_peak_at(peak_speed)
        with patch(
            "imap_l3_processing.swapi.l3a.science.solar_wind.proton."
            "initial_guess._gaussian_refine_bulk_speed_and_temperature",
            return_value=(peak_speed, INITIAL_TEMPERATURE_FLOOR_K),
        ) as patched_refine:
            calculate_initial_guess(ctx)

        temperature_seed_arg = patched_refine.call_args.args[3]
        self.assertEqual(temperature_seed_arg, INITIAL_TEMPERATURE_FLOOR_K)


class TestCalculateInitialGuessDirection(unittest.TestCase):
    """The initial velocity direction is anti-parallel to the chunk-mean spin
    axis. The spin axis is the unit-normalized mean of the body +Y axis (the
    second column of each SWAPI→RTN rotation matrix) across the chunk; the
    bulk velocity seed is `−|v_b|·ŝ`."""

    def test_initial_velocity_is_anti_parallel_to_spin_axis(self):
        # Build a ctx with our standard fixture and check that
        # `bulk_velocity_rtn` is anti-parallel to the chunk-mean spin axis.
        speed_grid = np.linspace(300.0, 600.0, 31)
        voltages = np.array(
            [_esa_voltage_for_proton_speed(s) for s in speed_grid]
        )
        count_rate = np.linspace(0.1, 0.5, len(voltages))
        peak_idx = len(speed_grid) // 2  # middle bin
        count_rate[peak_idx] = 100.0
        ctx = _build_proton_ctx(count_rate, voltages)

        # `_spin_rotation_matrices` aligns body +Y to -R̂ on every bin, so
        # the chunk-mean spin axis is exactly -R̂.
        expected_axis = np.array([-1.0, 0.0, 0.0])

        guess = calculate_initial_guess(ctx)
        v_unit = guess.bulk_velocity_rtn / np.linalg.norm(
            guess.bulk_velocity_rtn
        )
        np.testing.assert_allclose(v_unit, -expected_axis, atol=1e-12)

    def test_velocity_magnitude_matches_truth_bulk_speed(self):
        # On a noiseless ideal-rate spectrum, the Gaussian refine recovers
        # the truth bulk speed to high precision, so `|bulk_velocity_rtn|`
        # must equal ‖truth.bulk_velocity_rtn‖. The truth direction (+R̂)
        # is anti-parallel to this fixture's spin axis (-R̂).
        truth_bulk_speed = 450.0
        truth = SolarWindParams(
            density=5.0,
            bulk_velocity_rtn=np.array([truth_bulk_speed, 0.0, 0.0]),
            temperature=1.0e5,
            mass=PROTON_MASS_KG,
        )
        ctx = _make_synthetic_ctx_at_known_truth(truth)

        guess = calculate_initial_guess(ctx)
        np.testing.assert_allclose(
            np.linalg.norm(guess.bulk_velocity_rtn),
            truth_bulk_speed,
            rtol=2e-2,
        )


class TestCalculateInitialGuessDensity(unittest.TestCase):
    """The initial density is `optimal_density_scale(unit_ideal_rates,
    count_rate)` — the scale that minimizes residuals between the unit-density
    forward model and the observed counts (with deadtime correction)."""

    def test_density_is_optimal_scale_of_unit_density_forward_model(self):
        # Truth direction (+R̂) is anti-parallel to this fixture's spin axis (-R̂).
        truth = SolarWindParams(
            density=4.2,
            bulk_velocity_rtn=np.array([470.0, 0.0, 0.0]),
            temperature=1.2e5,
            mass=PROTON_MASS_KG,
        )
        ctx = _make_synthetic_ctx_at_known_truth(truth)

        guess = calculate_initial_guess(ctx)

        # Reproduce the function's density step independently: build a
        # unit-density forward-model evaluation at the *same* bulk-velocity
        # direction and temperature the function uses, then call
        # `optimal_density_scale`. The result must match `guess.density`.
        unit_density_params = SolarWindParams(
            density=1.0,
            bulk_velocity_rtn=guess.bulk_velocity_rtn,
            temperature=guess.temperature,
            mass=ctx.mass_kg,
        )
        unit_rates, _ = model_solar_wind_ideal_coincidence_rates(
            unit_density_params, ctx
        )
        expected_density = optimal_density_scale(unit_rates, ctx.count_rate)
        np.testing.assert_allclose(guess.density, expected_density, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
