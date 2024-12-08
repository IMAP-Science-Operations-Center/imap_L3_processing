import unittest
from datetime import datetime

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.glows.l3a.models import GlowsL2Data
from imap_processing.glows.l3a.utils import read_l2_glows_data
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    def test_reading_l2_glows_data_into_models(self):
        cdf = CDF(str(get_test_data_path('test_data/glows_l2_with_epoch.cdf')))

        glows_l2_data: GlowsL2Data = read_l2_glows_data(cdf)

        self.assertEqual((3600,), glows_l2_data.photon_flux.shape)
        self.assertEqual(29.0, glows_l2_data.photon_flux[0])
        self.assertEqual(29.0, glows_l2_data.photon_flux[-1])

        self.assertEqual((3600,), glows_l2_data.flux_uncertainties.shape)
        self.assertTrue(np.all(1 == glows_l2_data.flux_uncertainties))

        expected_start_time = datetime(1981, 9, 19, 17, 31, 21, 182398)
        self.assertEqual(expected_start_time, glows_l2_data.start_time)

        expected_end_time = datetime(1981, 9, 20, 15, 31, 21, 182398)
        self.assertEqual(expected_end_time, glows_l2_data.end_time)

        self.assertTrue(np.all(0.03333333333333333 == glows_l2_data.exposure_times))

        self.assertEqual((3600,), glows_l2_data.spin_angle.shape)
        self.assertEqual(0.000e+00, glows_l2_data.spin_angle[0])
        self.assertEqual(359.90000000000003, glows_l2_data.spin_angle[-1])

        self.assertEqual((4, 3600), glows_l2_data.histogram_flag_array.shape)
        self.assertEqual(bool, glows_l2_data.histogram_flag_array.dtype)

        expected_epoch = expected_start_time + 0.5 * (expected_end_time - expected_start_time)
        self.assertEqual(expected_epoch, glows_l2_data.epoch)

        self.assertEqual((3600,), glows_l2_data.ecliptic_lat.shape)
        self.assertEqual((3600,), glows_l2_data.ecliptic_lon.shape)
        self.assertEqual(0, glows_l2_data.ecliptic_lon[0])
        self.assertEqual(359.90000000000003, glows_l2_data.ecliptic_lon[-1])
        self.assertEqual(90, glows_l2_data.ecliptic_lat[0])
        self.assertEqual(-90, glows_l2_data.ecliptic_lat[-1])

    def test_fails_if_cdf_has_more_than_one_histogram(self):
        cdf = CDF(str(get_test_data_path('test_data/glows_l2_with_too_many_histograms.cdf')))

        with self.assertRaises(AssertionError) as cm:
            glows_l2_data: GlowsL2Data = read_l2_glows_data(cdf)
        self.assertEqual(("Level 2 file should have only one histogram",), cm.exception.args)


if __name__ == '__main__':
    unittest.main()
