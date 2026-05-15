import unittest

import numpy as np

from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import process_omni_alpha_param
from imap_l3_processing.glows.quality_flags import NOMINAL_ALPHA_PROTON_RATIO_VALUE


ALPHA_PARAM_SETTINGS = {
    "column_numbers": (0, 1, 2, 6, 9),
    "gap_marker": 9.999,
    "scale": False,
}


def _row(year: int, doy: int, hour: int, alpha: float) -> list[float]:
    return [year, doy, hour, 50.0, 5.0, 400.0, alpha, 0.1, 1.0, 0.001]


class TestProcessOmniAlphaParam(unittest.TestCase):
    def test_returns_zero_substitution_for_all_valid_alpha(self):
        omni_raw = np.array(
            [
                _row(2000, 1, 0, 0.04),
                _row(2000, 1, 1, 0.06),
                _row(2000, 28, 0, 0.05),
                _row(2000, 28, 1, 0.07),
            ]
        )

        cr_grid = np.array([1957, 1958])

        averaged, used_nominal_per_cr = process_omni_alpha_param(
            omni_raw, cr_grid, ALPHA_PARAM_SETTINGS
        )

        np.testing.assert_array_equal(used_nominal_per_cr, np.array([False, False]))
        np.testing.assert_array_almost_equal(averaged, np.array([0.05, 0.06]))

    def test_filters_invalid_samples_and_does_not_flag_cr_when_some_samples_remain(self):
        omni_raw = np.array(
            [
                _row(2000, 1, 0, 0.04),
                _row(2000, 1, 1, 9.999),
                _row(2000, 28, 0, 0.05),
                _row(2000, 28, 1, 0.07),
            ]
        )

        cr_grid = np.array([1957, 1958])

        averaged, used_nominal_per_cr = process_omni_alpha_param(
            omni_raw, cr_grid, ALPHA_PARAM_SETTINGS
        )

        np.testing.assert_array_equal(used_nominal_per_cr, np.array([False, False]))
        np.testing.assert_array_almost_equal(averaged, np.array([0.04, 0.06]))

    def test_fills_trailing_missing_crs_with_nominal_value(self):
        omni_raw = np.array(
            [
                _row(2000, 1, 0, 0.04),
                _row(2000, 1, 1, 0.06),
                _row(2000, 28, 0, 0.05),
                _row(2000, 28, 1, 0.07),
            ]
        )

        cr_grid = np.array([1957, 1958, 1959, 1960])

        averaged, used_nominal_per_cr = process_omni_alpha_param(
            omni_raw, cr_grid, ALPHA_PARAM_SETTINGS
        )

        np.testing.assert_array_equal(
            used_nominal_per_cr, np.array([False, False, True, True])
        )
        np.testing.assert_array_almost_equal(
            averaged,
            np.array(
                [0.05, 0.06, NOMINAL_ALPHA_PROTON_RATIO_VALUE, NOMINAL_ALPHA_PROTON_RATIO_VALUE]
            ),
        )

    def test_returns_all_nominal_when_param_cr_does_not_overlap_cr_grid(self):
        omni_raw = np.array(
            [
                _row(2000, 80, 0, 0.04),
                _row(2000, 82, 0, 0.06),
            ]
        )

        cr_grid = np.array([1957, 1958])

        averaged, used_nominal_per_cr = process_omni_alpha_param(
            omni_raw, cr_grid, ALPHA_PARAM_SETTINGS
        )

        np.testing.assert_array_equal(used_nominal_per_cr, np.array([True, True]))
        np.testing.assert_array_almost_equal(
            averaged,
            np.array([NOMINAL_ALPHA_PROTON_RATIO_VALUE, NOMINAL_ALPHA_PROTON_RATIO_VALUE]),
        )

    def test_raises_when_param_cr_starts_after_cr_grid_start(self):
        omni_raw = np.array(
            [
                _row(2000, 28, 0, 0.05),
                _row(2000, 28, 1, 0.07),
                _row(2000, 55, 0, 0.06),
                _row(2000, 56, 0, 0.08),
            ]
        )

        cr_grid = np.array([1957, 1958, 1959])

        with self.assertRaises(Exception) as ctx:
            process_omni_alpha_param(omni_raw, cr_grid, ALPHA_PARAM_SETTINGS)

        self.assertEqual(
            str(ctx.exception), "OMNI Error: not enough data for interpolation"
        )

    def test_smooths_interior_cr_gap_via_interpolation_and_does_not_flag(self):
        omni_raw = np.array(
            [
                _row(2000, 1, 0, 0.04),
                _row(2000, 1, 1, 0.06),
                _row(2000, 55, 0, 0.06),
                _row(2000, 56, 0, 0.08),
            ]
        )

        cr_grid = np.array([1957, 1958, 1959])

        averaged, used_nominal_per_cr = process_omni_alpha_param(
            omni_raw, cr_grid, ALPHA_PARAM_SETTINGS
        )

        np.testing.assert_array_equal(used_nominal_per_cr, np.array([False, False, False]))
        np.testing.assert_array_almost_equal(averaged, np.array([0.05, 0.06, 0.07]))


if __name__ == "__main__":
    unittest.main()
