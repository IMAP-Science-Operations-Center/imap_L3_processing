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

    def test_substitute_invalid_alpha_with_nominal_value_and_flag_corresponding_cr(self):
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

        np.testing.assert_array_equal(used_nominal_per_cr, np.array([True, False]))
        expected_cr_1957_alpha = (0.04 + NOMINAL_ALPHA_PROTON_RATIO_VALUE) / 2
        np.testing.assert_array_almost_equal(
            averaged, np.array([expected_cr_1957_alpha, 0.06])
        )


if __name__ == "__main__":
    unittest.main()
