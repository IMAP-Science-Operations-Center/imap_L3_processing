import unittest

import numpy as np
import numpy.testing as npt

from imap_l3_processing.constants import ALPHA_MASS_PER_CHARGE_M_P_PER_E
from imap_l3_processing.swapi.response.passband_grid import (
    _PASSBAND_BOUNDARY_THRESHOLD,
    _TARGET_ELEVATIONS,
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.response.speed_calculation import (
    SWAPI_K_FACTOR,
    esa_voltage_to_alpha_speed,
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


class TestSwapiResponseFromFiles(_RealResponseFixture):
    def test_loads_full_0_to_180_degree_azimuth_curve_at_0_1_deg_spacing(self):
        """CSV is spaced at 0.1° from 0° to 180° inclusive -> 1801 rows."""
        self.assertEqual(len(self.response.azimuthal_transmission), 1801)

    def test_sunglasses_attenuation(self):
        """For abs(azimuth)=0, attenuation=1/1000"""
        npt.assert_equal(self.response.azimuthal_transmission[0], 1/1000)

    def test_passband_fit_coefficients_loaded_for_oa_and_sg_regions(self):
        regions = self.response.passband_fit_coefficients.index.get_level_values(
            "region"
        )
        self.assertIn("OA", regions)
        self.assertIn("SG", regions)
        # Polynomial degree order (highest-first) is the convention `np.polyval`
        # consumes — coefficients are columns [2, 1, 0] for a quadratic fit.
        self.assertEqual(
            list(self.response.passband_fit_coefficients.columns), [2, 1, 0]
        )


class TestGetCentralEffectiveArea(_RealResponseFixture):
    def test_known_value_at_1kev_proton_beam(self):
        # Effective area at 1 keV proton ESA voltage from the CSV
        # `imap_swapi_central-effective-area_20260425_v001.csv` is 0.38 cm^2.
        npt.assert_approx_equal(
            self.response.get_central_effective_area(ESA_VOLTAGE_FOR_1KEV_PROTON),
            0.38,
            significant=3,
        )

    def test_uses_absolute_value_of_voltage(self):
        npt.assert_equal(
            self.response.get_central_effective_area(-ESA_VOLTAGE_FOR_1KEV_PROTON),
            self.response.get_central_effective_area(ESA_VOLTAGE_FOR_1KEV_PROTON),
        )


class TestCreatePassbandGridExtremeVoltages(_RealResponseFixture):
    """At voltages well outside the polynomial fit's calibration range, the
    polynomial extrapolation could go negative or blow up. The clamp inside
    `_get_passband_values` keeps values physical (non-negative, near unity)."""

    # Upper bound > 1.0 because the polynomial is fit to *un-normalized*
    # response data; normalization happens elsewhere. The fit can produce
    # values slightly above 1.0 at evaluation points between the fit nodes.
    PASSBAND_VALUE_UPPER_BOUND = 1.5

    def _assert_grid_value_bounds(self, esa_voltage):
        self.response.warm_cache([esa_voltage])
        for region in ("SG", "OA"):
            grid = self.response.create_passband_grid(esa_voltage, region)
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


class TestWarmCache(unittest.TestCase):
    """Tests here mutate the response instance, so build a fresh one per test."""

    def setUp(self):
        self.response = _load_response()

    def test_populates_cache_with_unique_finite_voltages(self):
        # Duplicates collapse, NaN/inf are skipped.
        voltages = np.array([100.0, 100.0, 200.0, 300.0, np.nan, np.inf])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._grid_cache.keys()), {100.0, 200.0, 300.0})
        self.assertEqual(len(self.response._grid_cache), 3)

    def test_warming_twice_at_same_voltage_returns_same_grid_values(self):
        # Two independently-warmed responses produce identical grids — the
        # build path is deterministic given the inputs.
        first_response = _load_response()
        second_response = _load_response()
        first_response.warm_cache([750.0])
        second_response.warm_cache([750.0])

        for region in ("SG", "OA"):
            with self.subTest(region=region):
                first_grid = first_response.create_passband_grid(750.0, region)
                second_grid = second_response.create_passband_grid(750.0, region)
                npt.assert_array_equal(first_grid.values, second_grid.values)

    def test_accepts_2d_voltage_array(self):
        # Higher-rank inputs are flattened via `.ravel()` before unique-ing.
        voltages = np.array([[100.0, 200.0], [200.0, 300.0]])
        self.response.warm_cache(voltages)
        self.assertEqual(set(self.response._grid_cache.keys()), {100.0, 200.0, 300.0})


class TestCreatePassbandGridRequiresWarmCache(unittest.TestCase):
    """`create_passband_grid` is the read path for cached PassbandGrids — it must
    *only* read the cache, never build. Building lazily would silently rebuild
    each grid in every forked worker (defeating the parent-process warm-up the
    caller relies on). When the requested voltage hasn't been warmed, the call
    must raise a clear KeyError that names the missing voltage."""

    def setUp(self):
        self.response = _load_response()

    def test_unwarmed_voltage_raises_keyerror_naming_voltage_and_remedy(self):
        with self.assertRaises(KeyError) as ctx:
            self.response.create_passband_grid(1234.0, "SG")
        msg = str(ctx.exception)
        self.assertIn("1234", msg)
        # The error message must point the caller at the fix.
        self.assertIn("warm_cache", msg)


class TestCentralSpeed(_RealResponseFixture):
    """`central_speed` returns the proton-frame speed at the ESA central energy
    for a species with given mass-per-charge. The mass-per-charge term divides
    under the sqrt, so passing 1.0 must reproduce the proton-only helper."""

    def test_proton_matches_esa_voltage_to_proton_speed(self):
        for v in [100.0, 1000.0, 4000.0]:
            with self.subTest(voltage=v):
                npt.assert_allclose(
                    self.response.central_speed(v, 1.0),
                    float(esa_voltage_to_proton_speed(v)),
                    rtol=1e-12,
                )

    def test_alpha_matches_esa_voltage_to_alpha_speed(self):
        # Alpha mass-per-charge is `(m_α/m_p) / (q_α/q_p) = (4×) / 2 = 2×` proton.
        for v in [500.0, 2000.0]:
            with self.subTest(voltage=v):
                npt.assert_allclose(
                    self.response.central_speed(v, ALPHA_MASS_PER_CHARGE_M_P_PER_E),
                    float(esa_voltage_to_alpha_speed(v)),
                    rtol=1e-12,
                )

    def test_uses_absolute_value_of_voltage(self):
        npt.assert_equal(
            self.response.central_speed(-1500.0, 1.0),
            self.response.central_speed(1500.0, 1.0),
        )

    def test_speed_scales_as_one_over_sqrt_mass_per_charge(self):
        # v ∝ 1/√(m/q) — doubling mass-per-charge halves speed by √2.
        v_proton = self.response.central_speed(1000.0, 1.0)
        v_alpha = self.response.central_speed(1000.0, 2.0)
        npt.assert_allclose(v_proton / v_alpha, np.sqrt(2.0), rtol=1e-12)


class TestResponseGridFieldOrder(unittest.TestCase):
    """`ResponseGrid` is a NamedTuple read by numba code that unpacks fields by
    position. Pin the field order so a rename or reorder doesn't silently break
    downstream JIT compilation."""

    def test_field_order_is_stable_for_numba_unpack(self):
        self.assertEqual(
            ResponseGrid._fields,
            (
                "sg_passband",
                "oa_passband",
                "central_speed",
                "central_effective_area",
                "azimuthal_transmission",
            ),
        )


class TestCreateResponseGrid(unittest.TestCase):
    """`create_response_grid` bundles the V-only `PassbandGrid` with V-and-species
    quantities (central speed and central effective area) into a `ResponseGrid`
    for the integrator. It memoises by `(voltage, species, ea_scale)` so repeated
    LM iterations at the same voltage skip the bundling work."""

    def setUp(self):
        self.response = _load_response()
        self.voltage = ESA_VOLTAGE_FOR_1KEV_PROTON
        self.response.warm_cache([self.voltage])

    def test_sg_and_oa_passband_fields_share_identity_with_cached_grids(self):
        rg = self.response.create_response_grid(self.voltage, 1.0)
        self.assertIs(
            rg.sg_passband, self.response.create_passband_grid(self.voltage, "SG")
        )
        self.assertIs(
            rg.oa_passband, self.response.create_passband_grid(self.voltage, "OA")
        )

    def test_central_speed_field_matches_central_speed_helper(self):
        rg = self.response.create_response_grid(self.voltage, 1.0)
        npt.assert_allclose(
            rg.central_speed, self.response.central_speed(self.voltage, 1.0)
        )

    def test_central_effective_area_field_matches_helper(self):
        rg = self.response.create_response_grid(self.voltage, 1.0)
        npt.assert_allclose(
            rg.central_effective_area,
            self.response.get_central_effective_area(self.voltage),
        )

    def test_azimuthal_transmission_grid_pass_through(self):
        rg = self.response.create_response_grid(self.voltage, 1.0)
        npt.assert_array_equal(
            rg.azimuthal_transmission.values, self.response.azimuthal_transmission
        )
        self.assertEqual(
            rg.azimuthal_transmission.spacing,
            self.response.AZIMUTHAL_TRANSMISSION_SPACING_DEG,
        )

    def test_effective_area_scale_multiplies_central_effective_area(self):
        # `central_effective_area_scale` is the species-specific efficiency
        # multiplier (e.g. alpha vs. proton CEM efficiency).
        unscaled = self.response.get_central_effective_area(self.voltage)
        rg_scaled = self.response.create_response_grid(
            self.voltage, 1.0, central_effective_area_scale=0.42
        )
        npt.assert_allclose(rg_scaled.central_effective_area, unscaled * 0.42)

    def test_central_speed_depends_on_species(self):
        # Different mass-per-charge → different central speed in the same grid.
        rg_proton = self.response.create_response_grid(self.voltage, 1.0)
        rg_alpha = self.response.create_response_grid(self.voltage, 2.0)
        self.assertGreater(rg_proton.central_speed, rg_alpha.central_speed)

    def test_repeat_call_returns_cached_object(self):
        # Regression: the cache key was previously a generator expression, so
        # every lookup missed and the dict grew unboundedly. Repeated calls
        # at the same args must return the *same object*, and the cache must
        # not grow.
        rg1 = self.response.create_response_grid(self.voltage, 1.0)
        rg2 = self.response.create_response_grid(self.voltage, 1.0)
        self.assertIs(rg1, rg2)
        self.assertEqual(len(self.response._response_grid_cache), 1)

    def test_cache_is_keyed_on_all_three_arguments(self):
        # A different value in any of (voltage, species, ea_scale) must produce
        # a distinct cache entry.
        rg_proton = self.response.create_response_grid(self.voltage, 1.0)
        rg_alpha = self.response.create_response_grid(self.voltage, 2.0)
        rg_scaled = self.response.create_response_grid(
            self.voltage, 1.0, central_effective_area_scale=0.5
        )
        self.assertIsNot(rg_proton, rg_alpha)
        self.assertIsNot(rg_proton, rg_scaled)
        self.assertEqual(len(self.response._response_grid_cache), 3)


class TestPassbandPolynomialBoundaries(unittest.TestCase):
    """Verify that the fitted polynomial boundaries bracket the region where the
    passband exceeds `_PASSBAND_BOUNDARY_THRESHOLD * max(grid)`, and that this
    cutoff is voltage-dependent."""

    @classmethod
    def setUpClass(cls):
        cls.response = _load_response()
        cls.response.warm_cache([ESA_VOLTAGE_FOR_2KEV_PROTON])
        cls.sg_grid = cls.response.create_passband_grid(
            ESA_VOLTAGE_FOR_2KEV_PROTON, "SG"
        )
        cls.oa_grid = cls.response.create_passband_grid(
            ESA_VOLTAGE_FOR_2KEV_PROTON, "OA"
        )

    def _check_boundaries(self, grid, region_label: str):
        """For each elevation, verify that grid values at speed ratios *outside*
        the integration window [min_ratio, max_ratio] returned by
        `speed_ratio_range_at_elevation` are at or below the cutoff. The
        bracketing-pair eval is designed to expansively cover the active region
        of the queried row, so any cell outside [lo, hi] is in the deep tail."""
        cutoff = _PASSBAND_BOUNDARY_THRESHOLD * float(grid.values.max())
        speed_ratios = (
            grid.min_speed_ratio
            + np.arange(grid.values.shape[1]) * grid.speed_ratio_spacing
        )

        for elevation in _TARGET_ELEVATIONS:
            row_idx = int(
                round((elevation - grid.min_elevation) / grid.elevation_spacing)
            )
            row = grid.values[row_idx]
            if not np.any(row > cutoff):
                continue  # no above-threshold passband at this elevation

            min_ratio, max_ratio = speed_ratio_range_at_elevation(
                grid, float(elevation)
            )

            outside = (speed_ratios < min_ratio - 1e-9) | (
                speed_ratios > max_ratio + 1e-9
            )
            with self.subTest(region=region_label, elevation=elevation):
                if np.any(outside):
                    self.assertLessEqual(row[outside].max(), cutoff + 1e-9)

    def test_oa_boundary_below_threshold(self):
        self._check_boundaries(self.oa_grid, region_label="OA")

    def test_sg_boundary_below_threshold(self):
        self._check_boundaries(self.sg_grid, region_label="SG")

    def test_boundaries_depend_on_voltage(self):
        """The min/max boundary arrays must differ between the low and high
        ends of each region's voltage range. The polynomial fit produces a
        different shape at different V."""
        for region in ("OA", "SG"):
            v_min, v_max = self.response.passband_esa_voltage_limits[region]
            low_voltage = v_min * 1.05
            high_voltage = v_max * 0.95
            self.response.warm_cache([low_voltage, high_voltage])
            grid_low = self.response.create_passband_grid(low_voltage, region)
            grid_high = self.response.create_passband_grid(high_voltage, region)

            for boundary_name in ("min_boundary", "max_boundary"):
                with self.subTest(region=region, boundary=boundary_name):
                    low = getattr(grid_low, boundary_name)
                    high = getattr(grid_high, boundary_name)
                    # The set of above-cutoff rows may shift with V — that's
                    # itself voltage-dependent variation, which shows up as a
                    # different NaN mask between the two arrays.
                    if not np.array_equal(np.isnan(low), np.isnan(high)):
                        continue
                    valid = ~np.isnan(low)
                    max_abs_diff = float(np.max(np.abs(low[valid] - high[valid])))
                    self.assertGreater(
                        max_abs_diff,
                        1e-6,
                        msg=f"{region}.{boundary_name} did not change between "
                        f"low-V and high-V grids (max|diff|={max_abs_diff:.2e})",
                    )

    def test_elevation_range_consistent_with_boundary(self):
        """The elevation range must bracket the first and last non-NaN
        rows of the boundary array (within one target-elevation spacing)."""
        spacing = float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0])
        for region, grid in [("OA", self.oa_grid), ("SG", self.sg_grid)]:
            with self.subTest(region=region):
                lower_active, upper_active = grid.elevation_range
                valid_row_indices = np.where(~np.isnan(grid.min_boundary))[0]
                first_boundary_elevation = (
                    grid.min_elevation
                    + valid_row_indices[0] * grid.elevation_spacing
                )
                last_boundary_elevation = (
                    grid.min_elevation
                    + valid_row_indices[-1] * grid.elevation_spacing
                )

                self.assertLessEqual(lower_active, first_boundary_elevation)
                self.assertGreaterEqual(upper_active, last_boundary_elevation)
                self.assertLessEqual(
                    first_boundary_elevation - lower_active, spacing + 1e-9
                )
                self.assertLessEqual(
                    upper_active - last_boundary_elevation, spacing + 1e-9
                )


if __name__ == "__main__":
    unittest.main()
