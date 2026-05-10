import unittest

import numpy as np
import pandas as pd

from imap_l3_processing.swapi.response.passband_grid import (
    _PASSBAND_BOUNDARY_THRESHOLD,
    _TARGET_ELEVATIONS,
    _TARGET_SPEED_RATIOS,
    build_passband_grid,
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.response.speed_calculation import SWAPI_K_FACTOR


def _gaussian_values_df(
    peak_elevation: float = 0.0,
    peak_speed_ratio: float = 1.0,
    sigma_el: float = 1.0,
    sigma_er: float = 0.1,
):
    """Synthetic per-region passband DataFrame indexed by `(energy_ratio, elevation)`,
    in the format `build_passband_grid` consumes. A single Gaussian bump centered
    at `(peak_elevation, peak_speed_ratio)` lets callers dial peak location and
    width to exercise specific branches. The sampling grid is deliberately wider
    than the target elevation range and at a different speed-ratio resolution so
    that the resampling step in `build_passband_grid` is non-trivial."""
    # Wider elevation range than target (-15..15) with extra rows on either side
    # that the resampler must drop. Elevation step matches target so every
    # target row finds an exact match in the DF index (the source uses `elev in
    # pivot.index`, an equality lookup).
    elevations = np.arange(-17.0, 17.0 + 0.5, 0.5)
    # Coarser, slightly wider speed-ratio grid than target — the resampler
    # interpolates between input nodes onto the denser target axis.
    speed_ratios = np.linspace(0.85, 1.15, 41)
    peak_er = SWAPI_K_FACTOR * peak_speed_ratio**2
    rows = [
        {
            "elevation": float(elev),
            "energy_ratio": float(SWAPI_K_FACTOR * sr**2),
            "value": float(
                # round so that there are true zeros like the real SIMION-derived passband
                np.exp(
                    -((elev - peak_elevation) ** 2) / (2.0 * sigma_el**2)
                    - ((SWAPI_K_FACTOR * sr**2 - peak_er) ** 2)
                    / (2.0 * sigma_er**2)
                )
            ),
        }
        for elev in elevations
        for sr in speed_ratios
    ]
    return pd.DataFrame(rows).set_index(["energy_ratio", "elevation"])


def _row_index_for_elevation(grid, elevation: float) -> int:
    """Index into `grid.values` for the row at the given target elevation."""
    return int(round((elevation - grid.min_elevation) / grid.elevation_spacing))


class TestBuildPassbandGrid(unittest.TestCase):
    """`build_passband_grid` takes a single `(energy_ratio, elevation)`-indexed
    DataFrame and returns a `PassbandGrid` containing the resampled value
    array, the per-elevation speed-ratio boundaries, and the elevation range."""

    @classmethod
    def setUpClass(cls):
        # Default-Gaussian grid shared by tests that just need a generic, well-
        # behaved passband. Built once per class because the arrays are read-only
        # and the build (~5 ms) is the dominant cost in this test class.
        cls.default_grid = build_passband_grid(_gaussian_values_df())

    def test_output_is_resampled_to_target_axes(self):
        """Regardless of the input DataFrame's sampling, the output is resampled
        onto the canonical `_TARGET_ELEVATIONS` × `_TARGET_SPEED_RATIOS` grid.
        Reconstruct the elevation and speed-ratio axes from `(min, spacing,
        shape)` and compare against the target arrays — any mismatch in shape,
        origin, or spacing surfaces here."""
        grid = self.default_grid
        n_el, n_sr = grid.values.shape
        elevations = grid.min_elevation + grid.elevation_spacing * np.arange(n_el)
        speed_ratios = grid.min_speed_ratio + grid.speed_ratio_spacing * np.arange(n_sr)
        np.testing.assert_allclose(elevations, _TARGET_ELEVATIONS)
        np.testing.assert_allclose(speed_ratios, _TARGET_SPEED_RATIOS)

    def test_boundary_marks_cutoff_crossing(self):
        """For every row inside the active elevation range, the row's value at
        `min_boundary[i]` and `max_boundary[i]` is exactly the cutoff (linear
        interpolation along the speed-ratio axis)."""
        grid = self.default_grid
        cutoff = _PASSBAND_BOUNDARY_THRESHOLD * grid.values.max()

        for i in range(grid.values.shape[0]):
            if np.isnan(grid.min_boundary[i]):
                continue
            with self.subTest(row=i):
                np.testing.assert_allclose(
                    np.interp(grid.min_boundary[i], _TARGET_SPEED_RATIOS, grid.values[i]),
                    cutoff,
                )
                np.testing.assert_allclose(
                    np.interp(grid.max_boundary[i], _TARGET_SPEED_RATIOS, grid.values[i]),
                    cutoff,
                )

    def test_elevation_range_brackets_peak(self):
        """Two grids built with peaks at different elevations must each have an
        elevation range that brackets *its own* peak and excludes the other's."""
        upper_peak = 10.0
        lower_peak = -10.0
        grid_high = build_passband_grid(_gaussian_values_df(peak_elevation=upper_peak))
        grid_low = build_passband_grid(_gaussian_values_df(peak_elevation=lower_peak))

        high_lo, high_hi = grid_high.elevation_range
        low_lo, low_hi = grid_low.elevation_range

        # peak in bounds
        self.assertLess(high_lo, upper_peak)
        self.assertLess(low_lo, lower_peak)
        self.assertGreater(high_hi, upper_peak)
        self.assertGreater(low_hi, lower_peak)

        # opposite peak out of either the lower or higher bound
        self.assertGreater(high_lo, lower_peak)
        self.assertLess(low_hi, upper_peak)

    def test_elevation_range_marks_cutoff_crossing(self):
        """Rounding the default Gaussian's tails to exactly zero pushes the
        outer rows below the cutoff. The elevation range edges must land
        exactly where `row_max` linearly interpolates to the cutoff."""
        df = _gaussian_values_df()
        df["value"] = df["value"].round(2)
        grid = build_passband_grid(df)

        cutoff = _PASSBAND_BOUNDARY_THRESHOLD * grid.values.max()
        row_max = grid.values.max(axis=1)

        lo, hi = grid.elevation_range
        np.testing.assert_allclose(np.interp(lo, _TARGET_ELEVATIONS, row_max), cutoff)
        np.testing.assert_allclose(np.interp(hi, _TARGET_ELEVATIONS, row_max), cutoff)
        # Range strictly tightened from the full target axis.
        self.assertGreater(lo, _TARGET_ELEVATIONS[0])
        self.assertLess(hi, _TARGET_ELEVATIONS[-1])

    def test_grid_is_normalized_so_on_axis_value_is_one(self):
        """`build_passband_grid` normalizes the grid so that
        `interpolate_passband(grid, 0.0, 1.0) == 1.0`."""
        df = _gaussian_values_df(peak_speed_ratio=1.05, sigma_er=0.05)
        grid = build_passband_grid(df)
        self.assertAlmostEqual(interpolate_passband(grid, 0.0, 1.0), 1.0)

    def test_speed_ratio_indexing_via_k_factor(self):
        """`_build_passband_array` converts `energy_ratio -> speed_ratio`
        via `sqrt(energy_ratio/k*)`. Energy ratio refers to E/|V| while speed ratio
        refers to v/v_0. v/v_0 = 1 corresponds to E/|V| = k*.
        An input whose peak happens to be at energy_ratio = k*
        and elevation = 0 (such as the default grid fixture)
        must produce a grid peaking at speed_ratio = 1.0."""
        grid = self.default_grid

        row_at_peak_elevation = grid.values[
            _row_index_for_elevation(grid, 0.0)
        ]
        peak_speed_ratio = _TARGET_SPEED_RATIOS[int(np.argmax(row_at_peak_elevation))]

        self.assertAlmostEqual(peak_speed_ratio, 1.0, places=2)
        self.assertGreater(row_at_peak_elevation.max(), 0.5)


class TestInterpolatePassband(unittest.TestCase):
    """Bilinear interpolation, zero outside the grid."""

    def setUp(self):
        self.grid = build_passband_grid(_gaussian_values_df(sigma_el=2.0))

    def test_matches_corner_weighted_sum(self):
        """`interpolate_passband` returns the bilinear-weighted sum of the four
            enclosing corner values."""
        cases = [
            # (elevation, speed_ratio, label)
            (0.0, 1.0, "on-node"),
            (1.25, 1.005, "cell center"),
            (1.1, 1.0014, "off-center"),
        ]
        for elevation, speed_ratio, label in cases:
            with self.subTest(label=label):
                i_float = (
                    elevation - self.grid.min_elevation
                ) / self.grid.elevation_spacing
                j_float = (
                    speed_ratio - self.grid.min_speed_ratio
                ) / self.grid.speed_ratio_spacing
                i_lo, j_lo = int(i_float), int(j_float)
                i_weight, j_weight = i_float - i_lo, j_float - j_lo

                corners = self.grid.values[
                    i_lo : i_lo + 2, j_lo : j_lo + 2
                ]
                weights = np.array(
                    [
                        [(1 - i_weight) * (1 - j_weight), (1 - i_weight) * j_weight],
                        [i_weight * (1 - j_weight), i_weight * j_weight],
                    ]
                )
                expected = float((corners * weights).sum())

                np.testing.assert_allclose(
                    interpolate_passband(self.grid, elevation, speed_ratio),
                    expected,
                    rtol=1e-12,
                )

    def test_returns_zero_outside_grid_in_either_axis(self):
        """Out-of-bounds query in either axis returns 0.0."""
        for el, sr, label in [
            (-100.0, 1.0, "below elevation"),
            (100.0, 1.0, "above elevation"),
            (0.0, 0.5, "below speed_ratio"),
            (0.0, 1.5, "above speed_ratio"),
        ]:
            with self.subTest(label=label):
                self.assertEqual(interpolate_passband(self.grid, el, sr), 0.0)


class TestSpeedRatioRangeAtElevation(unittest.TestCase):
    """`speed_ratio_range_at_elevation` returns the per-elevation min/max
    speed-ratio bounds, reading from the grid's `min_boundary` and
    `max_boundary` arrays."""

    def setUp(self):
        self.grid = build_passband_grid(_gaussian_values_df(peak_elevation=2.5))

    def test_returns_cutoff_crossing_for_on_grid_elevation(self):
        """At each on-grid elevation in the active range, the returned (lo, hi)
        lands near where the passband crosses the cutoff along the speed-ratio
        axis."""
        grid = build_passband_grid(_gaussian_values_df())
        cutoff = _PASSBAND_BOUNDARY_THRESHOLD * grid.values.max()
        active_lo, active_hi = grid.elevation_range
        for elevation in np.linspace(active_lo, active_hi, 5):
            with self.subTest(elevation=elevation):
                lo, hi = speed_ratio_range_at_elevation(grid, float(elevation))
                row = grid.values[_row_index_for_elevation(grid, float(elevation))]
                self.assertAlmostEqual(
                    np.interp(lo, _TARGET_SPEED_RATIOS, row), cutoff, places=2
                )
                self.assertAlmostEqual(
                    np.interp(hi, _TARGET_SPEED_RATIOS, row), cutoff, places=2
                )

    def test_returns_union_of_bracketing_rows_for_off_grid_elevation(self):
        """For an elevation between two grid rows, the returned (lo, hi)
        spans the union of the two rows' speed-ratio bounds."""
        lower_elevation = 0.0
        upper_elevation = 0.5
        between_elevation = 0.25
        lower_row = _row_index_for_elevation(self.grid, lower_elevation)
        upper_row = _row_index_for_elevation(self.grid, upper_elevation)
        expected_lo = min(
            self.grid.min_boundary[lower_row], self.grid.min_boundary[upper_row]
        )
        expected_hi = max(
            self.grid.max_boundary[lower_row], self.grid.max_boundary[upper_row]
        )
        lo, hi = speed_ratio_range_at_elevation(self.grid, between_elevation)
        self.assertAlmostEqual(lo, expected_lo)
        self.assertAlmostEqual(hi, expected_hi)

    def test_min_is_no_greater_than_max(self):
        """min <= max at every elevation inside the active range."""
        active_lo, active_hi = self.grid.elevation_range
        for elevation in np.linspace(active_lo, active_hi, 7):
            with self.subTest(elevation=elevation):
                lower, upper = speed_ratio_range_at_elevation(
                    self.grid, float(elevation)
                )
                self.assertLessEqual(lower, upper)

    def test_nan_row_handling_at_active_range_edges(self):
        """Outside the active passband the boundary arrays carry NaN. When the
        elevation lookup brackets a NaN row, behavior depends on how many of
        the two bracketing rows are NaN:

          - both NaN -> raises ValueError (caller failed to gate on the
            active elevation range)
          - exactly one NaN -> returns the valid row's bounds

        Both branches must be exercised at the top edge of the active range
        and at the bottom edge, where the asymmetric clamp gives
        (lower_row, upper_row) = (last, last) vs (0, 1) respectively.
        """
        grid = build_passband_grid(_gaussian_values_df())

        valid_rows = np.where(~np.isnan(grid.min_boundary))[0]
        first_valid = int(valid_rows[0])
        last_valid = int(valid_rows[-1])
        n_rows = grid.min_boundary.shape[0]
        # Sanity: the bottom-edge "both NaN" case needs rows 0 and 1 both NaN,
        # i.e. the active range must start at row 2 or later. The top-edge
        # case only needs the final row to be NaN.
        self.assertGreater(first_valid, 1)
        self.assertLess(last_valid, n_rows - 1)

        spacing = grid.elevation_spacing
        min_el = grid.min_elevation

        upper_edge_one_nan = min_el + spacing * (last_valid + 0.5)
        lower_edge_one_nan = min_el + spacing * (first_valid - 0.5)
        past_upper_edge = min_el + spacing * n_rows + 100.0
        past_lower_edge = min_el - 100.0

        with self.subTest(case="both NaN past upper edge"):
            with self.assertRaises(ValueError):
                speed_ratio_range_at_elevation(grid, float(past_upper_edge))

        with self.subTest(case="both NaN past lower edge"):
            with self.assertRaises(ValueError):
                speed_ratio_range_at_elevation(grid, float(past_lower_edge))

        with self.subTest(case="only upper row NaN at top of active range"):
            lo, hi = speed_ratio_range_at_elevation(grid, float(upper_edge_one_nan))
            self.assertAlmostEqual(lo, float(grid.min_boundary[last_valid]))
            self.assertAlmostEqual(hi, float(grid.max_boundary[last_valid]))

        with self.subTest(case="only lower row NaN at bottom of active range"):
            lo, hi = speed_ratio_range_at_elevation(grid, float(lower_edge_one_nan))
            self.assertAlmostEqual(lo, float(grid.min_boundary[first_valid]))
            self.assertAlmostEqual(hi, float(grid.max_boundary[first_valid]))


if __name__ == "__main__":
    unittest.main()
