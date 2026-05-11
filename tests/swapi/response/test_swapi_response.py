import unittest

import numpy as np
import numpy.testing as npt

from imap_l3_processing.constants import ALPHA_MASS_PER_CHARGE_M_P_PER_E
from imap_l3_processing.swapi.response.speed_calculation import (
    SWAPI_K_FACTOR,
    esa_voltage_to_proton_speed,
)
from imap_l3_processing.swapi.response.swapi_response import (
    ResponseGrid,
    SwapiResponse,
)
from tests.test_helpers import get_test_instrument_team_data_path

ESA_VOLTAGE_FOR_1KEV_PROTON = 1000.0 / SWAPI_K_FACTOR
ESA_VOLTAGE_FOR_2KEV_PROTON = 2000.0 / SWAPI_K_FACTOR

def _load_response() -> SwapiResponse:
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


class _RealResponseFixture(unittest.TestCase):
    """Base class loading a single shared `SwapiResponse` once per class."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()


class TestWarmCacheApi(unittest.TestCase):
    """Input-handling contract of `warm_cache`: dedup, NaN/inf skip, dimensionality,
    determinism. Each test mutates the response's `_grid_cache`, so a fresh
    instance is built per test."""

    def setUp(self):
        self.response = _load_response()

    def test_populates_cache_with_unique_finite_voltages(self):
        """Duplicates collapse, NaN/inf are skipped."""
        voltages = np.array([100.0, 100.0, 200.0, 300.0, np.nan, np.inf])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._passband_grid_cache.keys()), {100.0, 200.0, 300.0})
        self.assertEqual(len(self.response._passband_grid_cache), 3)

    def test_warming_twice_at_same_voltage_is_a_noop(self):
        """A second `warm_cache` at the same voltage keeps the existing
        PassbandGrid objects — it does not rebuild them."""
        self.response.warm_cache([750.0])
        first_grids = {
            region: self.response._passband_grid_cache[750.0][region] for region in ("SG", "OA")
        }
        self.response.warm_cache([750.0])

        for region in ("SG", "OA"):
            with self.subTest(region=region):
                self.assertIs(self.response._passband_grid_cache[750.0][region], first_grids[region])

    def test_accepts_2d_voltage_array(self):
        """Multidimensional voltage inputs are flattened."""
        voltages = np.array([[100.0, 200.0], [200.0, 300.0]])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._passband_grid_cache.keys()), {100.0, 200.0, 300.0})


class TestPassbandInterpolation(_RealResponseFixture):
    """Cover `_get_passband_values`: in-range it must evaluate
    `exp(polyval(coeffs_row, log(SWAPI_K_FACTOR * V)))` with coefficients read
    along the column axis (highest degree first); outside the per-region
    calibration window the voltage is clamped before evaluation, which keeps
    values physical (non-negative, near unity) even when the raw polynomial
    extrapolation would diverge."""

    # Upper bound > 1.0 because the polynomial is fit to *un-normalized*
    # response data; normalization happens elsewhere. The fit can produce
    # values slightly above 1.0 at evaluation points between the fit nodes.
    PASSBAND_VALUE_UPPER_BOUND = 1.5

    def test_values_match_manual_polyval(self):
        for region in ("OA", "SG"):
            with self.subTest(region=region):
                v_min, v_max = self.response._passband_esa_voltage_limits[region]
                voltage = 0.5 * (v_min + v_max)

                returned = self.response._get_passband_values(voltage, region)

                coeffs = self.response._passband_fit_coefficients.xs(
                    region, level="region"
                )
                log_beam_energy = np.log(SWAPI_K_FACTOR * voltage)
                n_degrees = coeffs.shape[1]
                degrees = np.arange(n_degrees - 1, -1, -1)
                expected_exponent = (coeffs.values * log_beam_energy ** degrees).sum(
                    axis=1
                )
                expected = np.exp(expected_exponent)

                npt.assert_allclose(
                    returned["value"].to_numpy(), expected, rtol=1e-12
                )

    def _assert_grid_value_bounds(self, esa_voltage):
        self.response.warm_cache([esa_voltage])
        cached = self.response._passband_grid_cache[self.response._cache_key(esa_voltage)]
        for region in ("SG", "OA"):
            grid = cached[region]
            self.assertGreaterEqual(
                grid.values.min(),
                0.0,
                msg=f"{region} passband has negative values at {esa_voltage} V",
            )
            self.assertLessEqual(
                grid.values.max(),
                self.PASSBAND_VALUE_UPPER_BOUND,
                msg=f"{region} passband exceeds bound at {esa_voltage} V",
            )

    def test_passband_grid_bounds_at_low_voltage(self):
        self._assert_grid_value_bounds(50.0)

    def test_passband_grid_bounds_at_high_voltage(self):
        self._assert_grid_value_bounds(20000.0)


class TestCreateResponseGrid(unittest.TestCase):
    """`create_response_grid` bundles the `PassbandGrid` with additional
    quantities (central speed and central effective area) into a `ResponseGrid`
    for the integrator. It caches by `(voltage, species, ea_scale)`."""

    def setUp(self):
        self.response = _load_response()
        self.voltage = ESA_VOLTAGE_FOR_1KEV_PROTON
        self.response.warm_cache([self.voltage])

    def test_raises_keyerror_when_voltage_not_warmed(self):
        """`get_response_grid` requires a prior `warm_cache` at the same V; the
        error message must name the voltage and the missing call so a caller
        can recover without reading the source."""
        fresh = _load_response()
        unwarmed_voltage = 12345.0
        with self.assertRaises(KeyError) as ctx:
            fresh.get_response_grid(unwarmed_voltage, 1.0)
        message = str(ctx.exception)
        self.assertIn(str(unwarmed_voltage), message)
        self.assertIn("warm_cache", message)

    def test_sg_and_oa_passband_fields_share_identity_with_cached_grids(self):
        rg = self.response.get_response_grid(self.voltage, 1.0)
        cached = self.response._passband_grid_cache[self.response._cache_key(self.voltage)]
        self.assertIs(rg.sg_passband, cached["SG"])
        self.assertIs(rg.oa_passband, cached["OA"])

    def test_central_speed_field_matches_esa_voltage_to_proton_speed(self):
        rg = self.response.get_response_grid(self.voltage, 1.0)
        npt.assert_allclose(
            rg.central_speed,
            float(esa_voltage_to_proton_speed(self.voltage)),
            rtol=1e-12,
        )

    def test_central_effective_area_field_matches_csv_at_1kev_proton(self):
        # Effective area at 1 keV proton ESA voltage from the CSV
        # `imap_swapi_central-effective-area_20260425_v001.csv` is 0.38 cm^2.
        rg = self.response.get_response_grid(self.voltage, 1.0)
        npt.assert_approx_equal(rg.central_effective_area, 0.38, significant=3)

    def test_azimuthal_transmission_grid_pass_through(self):
        rg = self.response.get_response_grid(self.voltage, 1.0)
        npt.assert_array_equal(
            rg.azimuthal_transmission.values, self.response._azimuthal_transmission
        )
        self.assertEqual(
            rg.azimuthal_transmission.spacing,
            self.response.AZIMUTHAL_TRANSMISSION_SPACING_DEG,
        )

    def test_effective_area_scale_multiplies_central_effective_area(self):
        """`central_effective_area_scale` is the species-specific efficiency
        multiplier (e.g. alpha vs. proton CEM efficiency)."""
        rg_unscaled = self.response.get_response_grid(self.voltage, 1.0)
        rg_scaled = self.response.get_response_grid(
            self.voltage, 1.0, central_effective_area_scale=0.42
        )
        npt.assert_allclose(
            rg_scaled.central_effective_area,
            rg_unscaled.central_effective_area * 0.42,
        )

    def test_central_speed_depends_on_species(self):
        """Different mass-per-charge → different central speed in the same grid.
        Reference values computed independently from CODATA constants at
        V = 1000/K (so K·e·V = 1 keV per unit charge):
          v_p = sqrt(2 · 1 keV · e / m_p) / 1e3 km/s
          v_α = sqrt(2 · 1 keV · 2e / m_α) / 1e3 km/s
        with m_p = 1.67262192595e-27 kg, m_α = 6.6446573450e-27 kg,
        e = 1.602176634e-19 C."""
        rg_proton = self.response.get_response_grid(self.voltage, 1.0)
        rg_alpha = self.response.get_response_grid(
            self.voltage, ALPHA_MASS_PER_CHARGE_M_P_PER_E
        )
        npt.assert_allclose(rg_proton.central_speed, 437.6947142244463, rtol=1e-12)
        npt.assert_allclose(rg_alpha.central_speed, 310.5624166704235, rtol=1e-12)

    def test_repeat_call_returns_cached_object(self):
        """Regression: the cache key was previously a generator expression, so
        every lookup missed and the dict grew unboundedly. Repeated calls at
        the same args must return the *same object*, and the cache must not
        grow."""
        rg1 = self.response.get_response_grid(self.voltage, 1.0)
        rg2 = self.response.get_response_grid(self.voltage, 1.0)
        self.assertIs(rg1, rg2)
        self.assertEqual(len(self.response._response_grid_cache), 1)

    def test_cache_is_keyed_on_all_three_arguments(self):
        """A different value in any of (voltage, species, ea_scale)
        must produce a distinct cache entry."""
        rg_proton = self.response.get_response_grid(self.voltage, 1.0)
        rg_alpha = self.response.get_response_grid(self.voltage, 2.0)
        rg_scaled = self.response.get_response_grid(
            self.voltage, 1.0, central_effective_area_scale=0.5
        )
        self.assertIsNot(rg_proton, rg_alpha)
        self.assertIsNot(rg_proton, rg_scaled)
        self.assertEqual(len(self.response._response_grid_cache), 3)


if __name__ == "__main__":
    unittest.main()
