import unittest
from datetime import datetime

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.ultra.l3.models import UltraGlowsL3eData, UltraL1CPSet
from tests.test_helpers import get_test_data_folder


class TestModels(unittest.TestCase):

    def test_glows_l3e_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l3e_survival_probabilities' / 'imap_glows_l3e_survival-probabilities-ultra_20250416_v001.cdf'

        actual = UltraGlowsL3eData.read_from_path(path_to_cdf)

        with CDF(str(path_to_cdf)) as expected:
            expected_epoch = datetime(2025, 4, 16, 12, 0)
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected['energy'][...], actual.energy)
            np.testing.assert_array_equal(expected['latitude'][...], actual.latitude)
            np.testing.assert_array_equal(expected['longitude'][...], actual.longitude)
            np.testing.assert_array_equal(expected['healpix_index'][...], actual.healpix_index)
            np.testing.assert_array_equal(expected['probability_of_survival'][...], actual.survival_probability)

    def test_ultra_l1c_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l1c_psets' / 'test_pset_nside1.cdf'

        actual = UltraL1CPSet.read_from_path(path_to_cdf)

        expected_epoch = datetime(2025, 9, 1, 0, 0)
        with CDF(str(path_to_cdf)) as expected:
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected["counts"][...], actual.counts)
            np.testing.assert_array_equal(expected["energy"][...], actual.energy)
            np.testing.assert_array_equal(expected["exposure_time"][...], actual.exposure)
            np.testing.assert_array_equal(expected["healpix_index"][...], actual.healpix_index)
            np.testing.assert_array_equal(expected["latitude"][...], actual.latitude)
            np.testing.assert_array_equal(expected["longitude"][...], actual.longitude)
            np.testing.assert_array_equal(expected["sensitivity"][...], actual.sensitivity)
