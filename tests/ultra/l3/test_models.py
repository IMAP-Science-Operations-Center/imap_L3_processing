import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.tests.ultra.data.mock_data import mock_l1c_pset_product_healpix
from spacepy.pycdf import CDF

from imap_l3_processing.ultra.l3.models import UltraGlowsL3eData, UltraL1CPSet
from tests.test_helpers import get_test_data_folder, run_local_data_path


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

    @patch('imap_processing.tests.ultra.data.mock_data.ensure_spice')
    def test_ultra_l1c_read_from_file(self, mock_ensure_spice):
        expected_epoch = datetime(2025, 9, 1, 0, 0)

        def fake_spice(tdb, et, tt):
            return expected_epoch.replace().timestamp()

        mock_ensure_spice.return_value = fake_spice
        pset_dataset = mock_l1c_pset_product_healpix(timestr=expected_epoch.isoformat())

        run_local_path = Path(run_local_data_path / 'ultra' / 'l1c_test_pset.cdf')
        run_local_path.unlink(missing_ok=True)

        with CDF(str(run_local_path), '') as cdf:
            for var in pset_dataset.variables:
                cdf[var] = pset_dataset.variables[var].values
                cdf[var].attrs['FILLVAL'] = -1e31

        actual = UltraL1CPSet.read_from_path(run_local_path)
        actual_epoch = datetime.fromtimestamp(actual.epoch / 1e9)

        with CDF(str(run_local_path)) as expected:
            self.assertEqual(expected_epoch, actual_epoch)
            np.testing.assert_array_equal(expected["counts"][...], actual.counts)
            np.testing.assert_array_equal(expected[CoordNames.ENERGY_ULTRA_L1C.value][...], actual.energy)
            np.testing.assert_array_equal(expected["exposure_factor"][...], actual.exposure)
            np.testing.assert_array_equal(expected[CoordNames.HEALPIX_INDEX.value][...], actual.healpix_index)
            np.testing.assert_array_equal(expected[CoordNames.ELEVATION_L1C.value][...], actual.latitude)
            np.testing.assert_array_equal(expected[CoordNames.AZIMUTH_L1C.value][...], actual.longitude)
            np.testing.assert_array_equal(expected["sensitivity"][...], actual.sensitivity)
