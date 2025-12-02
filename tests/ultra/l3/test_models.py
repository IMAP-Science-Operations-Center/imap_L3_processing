from datetime import datetime

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import ENERGY_VAR_NAME, PROBABILITY_OF_SURVIVAL_VAR_NAME, \
    HEALPIX_INDEX_VAR_NAME, EPOCH_CDF_VAR_NAME
from imap_l3_processing.ultra.l3.models import UltraGlowsL3eData, UltraL1CPSet
from tests.spice_test_case import SpiceTestCase
from tests.test_helpers import get_test_data_folder, with_tempdir, get_integration_test_data_path


class TestModels(SpiceTestCase):

    @with_tempdir
    def test_glows_l3e_read_from_file(self, temp_dir):
        path_to_cdf = temp_dir / 'imap_glows_l3e_survival-probability-ul_20250415-repoint01000_v001.cdf'

        rng = np.random.RandomState(42)
        expected_epoch = datetime(2025, 4, 15, 12, 0, 0, 1)
        expected_energy = np.arange(16)
        expected_healpix_index = np.arange(3072)
        expected_probability_of_survival = rng.random((1, 16, 3072))

        with CDF(str(path_to_cdf), masterpath='') as cdf:
            cdf[EPOCH_CDF_VAR_NAME] = [expected_epoch]
            cdf[EPOCH_CDF_VAR_NAME].attrs["FILLVAL"] = -38470932875435

            cdf[ENERGY_VAR_NAME] = expected_energy
            cdf[ENERGY_VAR_NAME].attrs["FILLVAL"] = -1e31

            cdf[HEALPIX_INDEX_VAR_NAME] = expected_healpix_index
            cdf[HEALPIX_INDEX_VAR_NAME].attrs["FILLVAL"] = 7340981273470912

            cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME] = expected_probability_of_survival
            cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME].attrs["FILLVAL"] = -1e31


        actual = UltraGlowsL3eData.read_from_path(path_to_cdf)

        self.assertEqual(expected_epoch, actual.epoch)
        self.assertEqual(1000, actual.repointing)
        np.testing.assert_array_equal(expected_energy, actual.energy)
        np.testing.assert_array_equal(expected_healpix_index, actual.healpix_index)
        np.testing.assert_array_equal(expected_probability_of_survival, actual.survival_probability)

    def test_ultra_l1c_read_from_file(self):
        expected_epoch = datetime(2025, 4, 15, 12)

        run_local_path = get_test_data_folder() / 'ultra' / 'fake_l1c_psets' / 'imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v010.cdf'

        actual = UltraL1CPSet.read_from_path(run_local_path)

        with CDF(str(run_local_path)) as expected:
            self.assertEqual(expected_epoch, actual.epoch)
            self.assertEqual(1, actual.repointing)
            np.testing.assert_array_equal(expected["counts"][...], actual.counts)
            np.testing.assert_array_equal(expected[CoordNames.ENERGY_ULTRA_L1C.value][...], actual.energy)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual.exposure)
            np.testing.assert_array_equal(expected["epoch_delta"][...], actual.epoch_delta)
            np.testing.assert_array_equal(expected[CoordNames.HEALPIX_INDEX.value][...], actual.healpix_index)
            np.testing.assert_array_equal(expected[CoordNames.ELEVATION_L1C.value][...], actual.latitude)
            np.testing.assert_array_equal(expected[CoordNames.AZIMUTH_L1C.value][...], actual.longitude)
            np.testing.assert_array_equal(expected["sensitivity"][...], actual.sensitivity)

    def test_ultra_l1c_read_from_file_handles_longitude_latitude_with_time_dimension(self):
        l1c_path = get_integration_test_data_path(
            "ultra/imap_ultra_l1c_45sensor-spacecraftpset_20250416-repoint00000_v000.cdf")

        actual = UltraL1CPSet.read_from_path(l1c_path)
        self.assertEqual(1, actual.longitude.ndim)
        self.assertEqual(1, actual.latitude.ndim)
