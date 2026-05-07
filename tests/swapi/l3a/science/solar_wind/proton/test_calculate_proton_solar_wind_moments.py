import math
import unittest

import scipy.optimize
from pathlib import Path

import numba
import numpy as np
from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    EV_TO_KELVIN,
    PROTON_MASS_KG,
    PROTON_CHARGE_COULOMBS,
    METERS_PER_KILOMETER,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.constants import SWAPI_LIVETIME_S
from imap_l3_processing.swapi.l3a.science.solar_wind import state
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    fit_solar_wind_proton_model,
)

from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    calculate_integral,
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams
from imap_l3_processing.swapi.response.passband_grid import interpolate_passband
from imap_l3_processing.swapi.response.response_grid import (
    interpolate_azimuthal_transmission,
)
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.response.response_grid import ResponseGrid
import pandas as pd
from imap_l3_processing.swapi.response.speed_calculation import (
    esa_voltage_to_proton_speed,
    SWAPI_K_FACTOR,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

_AZIMUTHAL_TRANSMISSION_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
)
_CENTRAL_EFFECTIVE_AREA_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
)
_PASSBAND_FIT_COEFFICIENTS_PATH = get_test_instrument_team_data_path(
    "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
)


def _load_swapi_response():
    return SwapiResponse.from_files(
        _AZIMUTHAL_TRANSMISSION_PATH,
        _CENTRAL_EFFECTIVE_AREA_PATH,
        _PASSBAND_FIT_COEFFICIENTS_PATH,
    )


_coverage_shared = {}


def _coverage_worker(noisy_rate):
    ctx = ProtonFitContext.from_l2_data(
        count_rate=noisy_rate,
        esa_voltage=_coverage_shared["voltages"],
        swapi_response=_coverage_shared["sr"],
        central_effective_area_scale=1.0,
        rotation_matrices=_coverage_shared["rot"],
    )
    return fit_solar_wind_proton_model(ctx)


_R_BASE_RTN_TO_SWAPI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_N_BINS = 71
_SWEEP_S = 12.0
_SPIN_S = 15.0
_DT_S = _SWEEP_S / 72


def _load_science_voltages():
    import spacepy.pycdf

    cdf_path = get_test_data_path("swapi/imap_swapi_l2_50-sweeps_20250606_v003.cdf")
    with spacepy.pycdf.CDF(str(cdf_path)) as cdf:
        esa_energy = cdf["esa_energy"][...]
    return esa_energy.mean(axis=0)[SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR


def _realistic_rotation_matrices(n_total, n_sweeps):
    """Returns SWAPI→RTN rotation matrices."""
    sweep_idx = np.arange(n_total) // _N_BINS
    bin_in_sweep = (np.arange(n_total) % _N_BINS) + 1
    times = sweep_idx * _SWEEP_S + bin_in_sweep * _DT_S
    alphas = 2.0 * np.pi * times / _SPIN_S
    R = np.empty((n_total, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        R[i] = (R_spin @ _R_BASE_RTN_TO_SWAPI).T
    return R


def _peak_voltage(bulk_speed_kms):
    """ESA voltage whose central speed equals bulk_speed_kms (inverse of esa_voltage_to_proton_speed)."""
    return float(
        PROTON_MASS_KG
        * (bulk_speed_kms * METERS_PER_KILOMETER) ** 2
        / (2 * SWAPI_K_FACTOR * PROTON_CHARGE_COULOMBS)
    )


def _thermal_speed(temperature_k):
    """Convert proton temperature in K to 1-D thermal speed in km/s."""
    return float(
        np.sqrt(BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * temperature_k / PROTON_MASS_KG)
        / METERS_PER_KILOMETER
    )


def _make_sw_params(
    density=5.0,
    temperature_k=10.0 * EV_TO_KELVIN,
    bulk_speed=450.0,
    bulk_azimuth=15.0,
    bulk_elevation=-5.0,
):
    """Build an SWParams with defaults that produce a nonzero integral for all three azimuth regions."""
    return SolarWindParams(
        density=density,
        bulk_speed=bulk_speed,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
        thermal_speed=_thermal_speed(temperature_k),
    )


def _build_proton_arrays(sr, voltages):
    """Build grids, central_speeds, central_effective_areas, az_trans, spacing for proton fits."""
    sr.warm_cache(voltages)
    grids = numba.typed.List([sr.create_passband_grid(v) for v in voltages])
    cs = np.array(
        [sr.central_speed(v, PROTON_MASS_PER_CHARGE_M_P_PER_E) for v in voltages]
    )
    cea = np.array([sr.get_central_effective_area(v) for v in voltages])
    at = np.asarray(sr.azimuthal_transmission, dtype=float)
    ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)
    return grids, cs, cea, at, ats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeadtimeFactor(unittest.TestCase):
    """Verify deadtime_factor against a known (true rate, measured rate) pair."""

    def test_example(self):
        # At g=35.000 kHz measured, n=35.226 kHz true.
        # C_obs = C_true * deadtime_factor(C_true) should recover ~35000 Hz.
        self.assertAlmostEqual(35226 * deadtime_factor(35226), 35000, delta=1)


# ---------------------------------------------------------------------------


class TestEsaVoltageToProtonSpeed(unittest.TestCase):
    """Verify the ESA voltage → proton speed conversion: known value, negative-voltage symmetry, array output."""

    def test_known_value(self):
        # At V = 1000 V, E = k* * V = 1.89 keV
        # v = sqrt(2 * 1890 eV * e / m_p) = 601.730748 km/s (independently computed)
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(1000.0), 601.730748, rtol=1e-3
        )

    def test_uses_absolute_value_of_voltage(self):
        np.testing.assert_allclose(
            esa_voltage_to_proton_speed(-529.0), esa_voltage_to_proton_speed(529.0)
        )

    def test_vectorized(self):
        speeds = esa_voltage_to_proton_speed(np.array([500.0, 1000.0, 2000.0]))
        self.assertEqual(speeds.shape, (3,))
        self.assertTrue(np.all(np.diff(speeds) > 0))
        np.testing.assert_allclose(speeds[1], 601.730748, rtol=1e-3)


class TestInterpolatePassband(unittest.TestCase):
    """Verify interpolate_passband returns sensible in-bounds values and 0 out-of-bounds,
    and that SG and OA grids differ at the same point."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        sr.warm_cache([_peak_voltage(450.0)])
        cls.grid = sr.create_passband_grid(_peak_voltage(450.0))

    def test_central_value_is_one_for_normalized_grid(self):
        # At elevation=0 and speed_ratio=1.0 (central speed), OA passband should be ~1 (peak)
        val = interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=1.0)
        self.assertGreater(val, 0.5)

    def test_out_of_bounds_elevation_returns_zero(self):
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=100.0, speed_ratio=1.0),
            0.0,
        )
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=-100.0, speed_ratio=1.0),
            0.0,
        )

    def test_out_of_bounds_speed_returns_zero(self):
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=0.0), 0.0
        )
        np.testing.assert_equal(
            interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=10.0), 0.0
        )

    def test_sg_and_oa_are_different(self):
        sg = interpolate_passband(self.grid, True, elevation=0.0, speed_ratio=1.0)
        oa = interpolate_passband(self.grid, False, elevation=0.0, speed_ratio=1.0)
        self.assertFalse(np.isclose(sg, oa))


class TestInterpolateTransmission(unittest.TestCase):
    """Verify interpolate_azimuthal_transmission against known calibration values, symmetry, and periodicity."""

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        cls.at = np.asarray(sr.azimuthal_transmission, dtype=float)
        cls.ats = float(sr.AZIMUTHAL_TRANSMISSION_SPACING_DEG)

    def test_sunglasses_region_near_zero(self):
        # Sunglasses region |phi| < 9° has ~1e-3 transmission
        val = interpolate_azimuthal_transmission(self.at, self.ats, 0.0)
        self.assertLess(val, 0.01)

    def test_open_aperture_region_near_one(self):
        # Open aperture region has transmission close to 1
        val = interpolate_azimuthal_transmission(self.at, self.ats, 90.0)
        self.assertGreater(val, 0.5)

    def test_periodic_wraparound(self):
        # 170° and −170° should give the same value (both near 180°)
        v1 = interpolate_azimuthal_transmission(self.at, self.ats, 170.0)
        v2 = interpolate_azimuthal_transmission(self.at, self.ats, -170.0)
        np.testing.assert_allclose(v1, v2, rtol=1e-6)

    def test_symmetric_about_zero(self):
        # Transmission is symmetric: T(phi) = T(-phi)
        for phi in [5.0, 45.0, 90.0, 120.0]:
            v_pos = interpolate_azimuthal_transmission(self.at, self.ats, phi)
            v_neg = interpolate_azimuthal_transmission(self.at, self.ats, -phi)
            np.testing.assert_allclose(v_pos, v_neg, rtol=1e-6, err_msg=f"phi={phi}")


def _spin_rotation_matrices(n, spin_period_s=15.0, dt_s=0.145):
    """Realistic SWAPI geometry: spin axis = boresight (+Y_SWAPI = -R_RTN).

    Builds RTN→SWAPI as `R_spin_around_Y(2*pi*t/T_spin) @ R_base` (R_base maps
    nominal anti-sunward bulk (R_RTN) to -Y_SWAPI so phi=theta=0 at zero spin
    phase), then transposes to return SWAPI→RTN.
    """
    times = np.arange(n) * dt_s
    alphas = 2.0 * np.pi * times / spin_period_s
    R = np.empty((n, 3, 3))
    for i, a in enumerate(alphas):
        c, s = np.cos(a), np.sin(a)
        R_spin = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        R[i] = (R_spin @ _R_BASE_RTN_TO_SWAPI).T
    return R


class TestInterpolateTransmissionBoundary(unittest.TestCase):
    """Verify interpolate_azimuthal_transmission returns 0 when both interpolation indices clamp to the
    same out-of-bounds entry (weights cancel). Uses a 3-element array to trigger easily."""

    def setUp(self):
        from imap_l3_processing.swapi.response.passband_grid import PassbandGrid

        zero_grid = np.zeros((23, 101), dtype=np.float64)
        boundary = np.array([[0.0], [0.95]])
        self.at = np.array([0.001, 0.5, 1.0])
        self.ats = 1.0
        self.grid = PassbandGrid(
            min_elevation=-12.0,
            elevation_spacing=1.0,
            min_speed_ratio=0.9,
            speed_ratio_spacing=0.002,
            values_sunglasses=zero_grid,
            values_open_aperture=zero_grid,
            min_OA_boundary=boundary,
            max_OA_boundary=boundary,
            min_SG_boundary=boundary,
            max_SG_boundary=boundary,
            oa_elevation_range=(-12.0, 10.5),
            sg_elevation_range=(-10.5, 7.0),
        )

    def test_azimuth_beyond_array_returns_zero(self):
        # When both interpolation indices clamp to the same out-of-bounds entry,
        # the symmetric weights cancel and transmission is 0. Spot-check at both
        # just-beyond (i_upper clamps) and far-beyond (i_lower clamps) cases.
        for azimuth in (2.5, 170.0):
            with self.subTest(azimuth=azimuth):
                val = interpolate_azimuthal_transmission(self.at, self.ats, azimuth)
                self.assertEqual(val, 0.0)


class TestOptimalDensityScale(unittest.TestCase):
    def _scipy_minimize(self, predicted, observed):
        result = scipy.optimize.minimize_scalar(
            lambda s: np.sum((s * predicted - observed) ** 2)
        )
        return result.x

    def test_matches_scipy_random_vectors(self):
        rng = np.random.default_rng(0)
        predicted = rng.uniform(0.1, 100.0, size=50)
        observed = rng.uniform(0.1, 100.0, size=50)
        # rtol=1e-4 reflects scipy.optimize.minimize_scalar's Brent's-method
        # convergence (~1e-5); the analytic closed form is exact to machine
        # precision, so this asserts they agree to scipy's tolerance.
        np.testing.assert_allclose(
            optimal_density_scale(predicted, observed),
            self._scipy_minimize(predicted, observed),
            rtol=1e-4,
        )

    def test_matches_scipy_when_scale_less_than_one(self):
        rng = np.random.default_rng(1)
        predicted = rng.uniform(10.0, 200.0, size=30)
        observed = predicted * 0.3 + rng.normal(0, 0.5, size=30)
        np.testing.assert_allclose(
            optimal_density_scale(predicted, observed),
            self._scipy_minimize(predicted, observed),
            rtol=1e-4,
        )

    def test_zero_predicted_returns_one(self):
        self.assertEqual(optimal_density_scale(np.zeros(5), np.ones(5)), 1.0)


class TestAnalyticJacobian(unittest.TestCase):
    """Verify the analytic Jacobian returned by `model_solar_wind_ideal_coincidence_rates`
    matches a central-difference numerical Jacobian.

    State-vector layout: [ln n, ln T, v_R, v_T, v_N]. The forward model returns
    (rates, jac) where jac has columns [d/d ln n, d/d ln T, d/d v_R, d/d v_T,
    d/d v_N].
    """

    @classmethod
    def setUpClass(cls):
        sr = _load_swapi_response()
        voltages = _load_science_voltages()
        n_sweeps = 2
        all_voltages = np.tile(voltages, n_sweeps)
        rotation_matrices = _realistic_rotation_matrices(n_sweeps * _N_BINS, n_sweeps)
        sr.warm_cache(all_voltages)

        cls.ctx = build_solar_wind_fit_context(
            count_rate=np.ones_like(all_voltages),
            esa_voltage=all_voltages,
            swapi_response=sr,
            central_effective_area_scale=1.0,
            rotation_matrices=rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )
        cls.params = SolarWindParams(
            density=5.0,
            bulk_velocity_rtn=np.array([450.0, 20.0, 5.0]),
            temperature=1e5,
            mass=PROTON_MASS_KG,
        )

    def test_analytic_jacobian_matches_finite_difference(self):
        rates, analytic_jac = model_solar_wind_ideal_coincidence_rates(
            self.params, self.ctx
        )

        # Central-difference numerical Jacobian over the 5 state-vector
        # parameters. For ln n / ln T a 1e-4 step is fine; for velocities
        # 1e-2 km/s gives a usable difference without losing precision.
        state = self.params.to_vector()
        steps = np.array([1e-4, 1e-4, 1e-2, 1e-2, 1e-2])

        numerical_jac = np.empty_like(analytic_jac)
        for j in range(5):
            h = steps[j]

            state_plus = state.copy()
            state_plus[j] += h
            params_plus = SolarWindParams.from_vector(state_plus, mass=self.params.mass)
            rates_plus, _ = model_solar_wind_ideal_coincidence_rates(
                params_plus, self.ctx
            )

            state_minus = state.copy()
            state_minus[j] -= h
            params_minus = SolarWindParams.from_vector(
                state_minus, mass=self.params.mass
            )
            rates_minus, _ = model_solar_wind_ideal_coincidence_rates(
                params_minus, self.ctx
            )

            numerical_jac[:, j] = (rates_plus - rates_minus) / (2.0 * h)

        # Restrict comparison to bins where the model rate is physically
        # significant (>= 1 Hz); tiny rates are dominated by truncation /
        # quadrature noise and don't constrain the Jacobian usefully.
        mask = rates > 1.0
        self.assertGreater(
            int(np.sum(mask)),
            0,
            "Fixture produced no bins with rate > 1 Hz; check setUpClass.",
        )

        # Compare each Jacobian column normalized by its column scale: per-bin
        # relative error blows up where d C / d p crosses zero (the "1.5 σ²"
        # cancellation for d/d ln T, geometry-driven nulls for velocity columns).
        param_names = ["ln n", "ln T", "v_R", "v_T", "v_N"]
        for j, name in enumerate(param_names):
            col_scale = float(np.max(np.abs(analytic_jac[mask, j])))
            self.assertGreater(
                col_scale, 0.0, f"Analytic column for {name} is identically zero."
            )
            max_abs_err = float(
                np.max(np.abs(numerical_jac[mask, j] - analytic_jac[mask, j]))
            )
            normalized_err = max_abs_err / col_scale
            self.assertLess(
                normalized_err,
                5e-3,
                msg=(
                    f"Analytic vs numerical Jacobian disagree for d/d {name}: "
                    f"max abs err = {max_abs_err:.3e}, column scale = {col_scale:.3e}, "
                    f"normalized = {normalized_err:.3e} (>{5e-3:.0e})."
                ),
            )


if __name__ == "__main__":
    unittest.main()
