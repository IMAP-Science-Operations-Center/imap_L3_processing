"""Tests for `solar_wind/fit_context.py`.

Covers `SolarWindFitContext.subset`, `average_spin_axis_rtn`, and the
voltage-filter logic in `build_solar_wind_fit_context`.

No SPICE required.  `build_solar_wind_fit_context` needs a `SwapiResponse`
object to build `response_grids`; for voltage-filter tests we mock the response.
For subset tests we build a real context via the swapi_response test helper (the
cache means the expensive parse happens only once per test run).
"""

import unittest

import numpy as np

from imap_l3_processing.constants import (
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import (
    average_spin_axis_rtn,
)
from tests.swapi._swapi_test_helpers import swapi_response as _load_swapi_response


def _make_context(n: int) -> SolarWindFitContext:
    """Build a real SolarWindFitContext with `n` sweeps using the cached SwapiResponse.

    Uses n distinct voltages spread around the solar-wind peak to get valid grids.
    `response_grids[i]` objects are distinct — identity verified by position rather
    than content (they are ResponseGrid NamedTuples and not singletons).
    """
    sr = _load_swapi_response()
    # Use voltages in a range that the calibration files cover.
    esa_voltages = np.linspace(800.0, 1400.0, n)
    sr.warm_cache(esa_voltages)
    count_rate = np.arange(n, dtype=float)
    rotation_matrices = np.arange(n * 9, dtype=float).reshape(n, 3, 3)
    return build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltages,
        swapi_response=sr,
        central_effective_area_scale=1.0,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )


class TestSolarWindFitContextSubset(unittest.TestCase):
    """SolarWindFitContext.subset returns a context sliced to the given indices."""

    def test_subset_selects_correct_rows_and_grids(self):
        """Subsetting picks the right rows from count_rate, esa_voltage,
        rotation_matrices, and response_grids."""
        ctx = _make_context(8)
        indices = np.array([1, 4])
        sub = ctx.subset(indices)

        np.testing.assert_array_equal(sub.count_rate, ctx.count_rate[indices])
        np.testing.assert_array_equal(sub.esa_voltage, ctx.esa_voltage[indices])
        np.testing.assert_array_equal(
            sub.rotation_matrices, ctx.rotation_matrices[indices]
        )
        for sub_i, orig_i in enumerate(indices):
            self.assertAlmostEqual(
                sub.response_grids[sub_i].central_speed,
                ctx.response_grids[orig_i].central_speed,
                places=10,
            )

    def test_subset_empty_indices_produces_zero_length_arrays(self):
        """An empty index list gives per-measurement arrays of length 0 without error."""
        ctx = _make_context(5)
        sub = ctx.subset([])

        self.assertEqual(len(sub.count_rate), 0)
        self.assertEqual(len(sub.esa_voltage), 0)
        self.assertEqual(len(sub.response_grids), 0)
        self.assertEqual(sub.rotation_matrices.shape[0], 0)


class TestAverageSpinAxisRtn(unittest.TestCase):
    """average_spin_axis_rtn returns a unit-normalized mean of the per-sweep +Y axes.

    The second row (index 1) of each 3×3 rotation matrix is extracted and averaged.
    """

    def test_result_is_unit_norm(self):
        """Result is always unit-normalized regardless of the input stack."""
        rng = np.random.default_rng(42)
        # Build random (non-unitary) rotation matrices — normalization must still hold.
        rotation_matrices = rng.standard_normal((7, 3, 3))
        result = average_spin_axis_rtn(rotation_matrices)

        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=12)

    def test_nominal_minus_r_direction(self):
        """SWAPI nominal attitude: +Y_SWAPI ≈ -R̂_RTN.

        A SWAPI→RTN rotation matrix whose column 1 is [-1, 0, 0] (the -R
        direction in RTN) corresponds to the boresight pointing at -R. The
        average of such a stack should be [-1, 0, 0] normalized.
        """
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        rotation_matrices = np.tile(R, (6, 1, 1))
        result = average_spin_axis_rtn(rotation_matrices)

        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-12)


class TestBuildSolarWindFitContextVoltageFilter(unittest.TestCase):
    """build_solar_wind_fit_context drops non-positive and non-finite voltages.

    Uses the real SwapiResponse (via the cached test helper) with voltages inside
    the calibration-table range so the grids build successfully.  The filter is
    tested purely by inspecting output lengths and retained voltages.
    """

    @classmethod
    def setUpClass(cls):
        """Load the SwapiResponse once and warm the cache for all test voltages."""
        cls.sr = _load_swapi_response()
        # Warm cache for all valid voltages used across the tests.
        cls.sr.warm_cache([500.0, 400.0, 300.0, 200.0, 800.0, 600.0])

    def _build(self, esa_voltage, count_rate=None, rotation_matrices=None):
        n = len(esa_voltage)
        if count_rate is None:
            count_rate = np.ones(n)
        if rotation_matrices is None:
            rotation_matrices = np.tile(np.eye(3), (n, 1, 1))
        return build_solar_wind_fit_context(
            count_rate=count_rate,
            esa_voltage=np.array(esa_voltage, dtype=float),
            swapi_response=self.sr,
            central_effective_area_scale=1.0,
            rotation_matrices=rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )

    def test_drops_nan_and_zero_voltages(self):
        """NaN and zero voltages are filtered; only 3 valid sweeps survive."""
        # Input: [500, 0, 300, nan, 200] → keep [500, 300, 200]
        ctx = self._build([500.0, 0.0, 300.0, np.nan, 200.0])

        self.assertEqual(len(ctx.esa_voltage), 3)
        self.assertEqual(len(ctx.count_rate), 3)
        self.assertEqual(ctx.rotation_matrices.shape[0], 3)
        self.assertEqual(len(ctx.response_grids), 3)

    def test_drops_negative_voltages(self):
        """Negative voltages are dropped (condition: voltage > 0, so negative fails)."""
        # Input: [400, -100, 300] → keep [400, 300] (rows 0, 2)
        ctx = self._build([400.0, -100.0, 300.0])

        self.assertEqual(len(ctx.esa_voltage), 2)
        np.testing.assert_array_equal(ctx.esa_voltage, [400.0, 300.0])

    def test_all_valid_voltages_unchanged(self):
        """When all voltages are valid, context keeps all sweeps."""
        ctx = self._build([500.0, 400.0, 300.0, 200.0])
        self.assertEqual(len(ctx.esa_voltage), 4)


if __name__ == "__main__":
    unittest.main()
