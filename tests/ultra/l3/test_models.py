from datetime import datetime

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import ENERGY_VAR_NAME, PROBABILITY_OF_SURVIVAL_VAR_NAME, \
    LATITUDE_VAR_NAME, LONGITUDE_VAR_NAME, HEALPIX_INDEX_VAR_NAME
from imap_l3_processing.ultra.l3.models import UltraGlowsL3eData, UltraL1CPSet
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import get_test_data_folder


class TestModels(SpiceTestCase):

    def test_glows_l3e_read_from_file(self):
        path_to_cdf = get_test_data_folder() / 'ultra' / 'fake_l3e_survival_probabilities' / 'imap_glows_l3e_survival-probabilities-ultra_20250416_v001.cdf'

        actual = UltraGlowsL3eData.read_from_path(path_to_cdf)

        with CDF(str(path_to_cdf)) as expected:
            expected_epoch = datetime(2025, 4, 16, 12, 0)
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected[ENERGY_VAR_NAME][...], actual.energy)
            np.testing.assert_array_equal(expected[LATITUDE_VAR_NAME][...], actual.latitude)
            np.testing.assert_array_equal(expected[LONGITUDE_VAR_NAME][...], actual.longitude)
            np.testing.assert_array_equal(expected[HEALPIX_INDEX_VAR_NAME][...], actual.healpix_index)
            np.testing.assert_array_equal(expected[PROBABILITY_OF_SURVIVAL_VAR_NAME][...], actual.survival_probability)

    def test_ultra_l1c_read_from_file(self):
        expected_epoch = datetime(2025, 4, 15, 0)

        run_local_path = get_test_data_folder() / 'ultra' / 'fake_l1c_psets' / 'imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v010.cdf'

        actual = UltraL1CPSet.read_from_path(run_local_path)

        with CDF(str(run_local_path)) as expected:
            self.assertEqual(expected_epoch, actual.epoch)
            np.testing.assert_array_equal(expected["counts"][...], actual.counts)
            np.testing.assert_array_equal(expected[CoordNames.ENERGY_ULTRA.value][...], actual.energy)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual.exposure)
            np.testing.assert_array_equal(expected[CoordNames.HEALPIX_INDEX.value][...], actual.healpix_index)
            np.testing.assert_array_equal(expected[CoordNames.ELEVATION_L1C.value][...], actual.latitude)
            np.testing.assert_array_equal(expected[CoordNames.AZIMUTH_L1C.value][...], actual.longitude)
            np.testing.assert_array_equal(expected["sensitivity"][...], actual.sensitivity)
