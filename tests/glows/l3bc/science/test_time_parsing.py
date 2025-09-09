import unittest

import numpy as np
from astropy.time import Time, conf

from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import time_from_yday as time_from_yday_l3bc
from imap_l3_processing.glows.l3d.science.toolkit.funcs import time_from_yday as time_from_yday_l3d


class TestTimeParsing(unittest.TestCase):
    def test_works_with_fast_parser(self):
        time_rows = np.array([
            [2000.0, 5.0,1.0],
            [2000.0, 51.0, 12.0],
            [2000.0, 365.0, 23.0],
        ])
        with conf.set_temp("use_fast_parser", "force"):
            try:
                result_l3bc = time_from_yday_l3bc(time_rows)
                result_l3d = time_from_yday_l3d(time_rows)
            except ValueError as e:
                self.fail('Fast parser should not crash, check formatting of date string.')

        np.testing.assert_array_equal(result_l3bc, Time([
            "2000-01-05T01:00",
            "2000-02-20T12:00",
            "2000-12-30T23:00",
        ]))

        np.testing.assert_array_equal(result_l3d, Time([
            "2000-01-05T01:00",
            "2000-02-20T12:00",
            "2000-12-30T23:00",
        ]))


if __name__ == '__main__':
    unittest.main()
