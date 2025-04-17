import os
import unittest
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from tests.test_helpers import get_test_data_path


class TestCodiceLoL3aDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.download_dependency_from_path")
    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL3aDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_utils_download):
        processing_input_collection = ProcessingInputCollection()

        expected_codice_science_file_download_path = "imap/codice/l2/2010/01/imap_codice_l2_sectored-intensities_20100105_v010.cdf"
        expected_mass_per_charge_file_download_path = "imap/ancillary/codice/imap_codice_mass-per-charge-lookup_20100105_v001.csv"
        expected_esa_step_file_download_path = "imap/ancillary/codice/imap_codice_esa-step-lookup_20100105_v001.csv"

        codice_sectored_intensity_input_file_name = "imap_codice_l2_sectored-intensities_20100105_v010.cdf"
        mass_per_charge_lookup_file = "imap_codice_mass-per-charge-lookup_20100105_v001.csv"
        esa_step_lookup_file = "imap_codice_esa-step-lookup_20100105_v001.csv"

        science_input = ScienceInput(codice_sectored_intensity_input_file_name)
        ancillary_input_1 = AncillaryInput(mass_per_charge_lookup_file)
        ancillary_input_2 = AncillaryInput(esa_step_lookup_file)

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        processing_input_collection.add(
            [science_input, ancillary_input_1, ancillary_input_2, non_codice_ancillary_input, non_codice_science_input,
             non_l2_codice_science_input])

        codice_l3_dependencies = CodiceLoL3aDependencies.fetch_dependencies(processing_input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_science_path = data_dir / expected_codice_science_file_download_path
        expected_download_ancillary_path_1 = data_dir / expected_mass_per_charge_file_download_path
        expected_download_ancillary_path_2 = data_dir / expected_esa_step_file_download_path

        mock_utils_download.assert_has_calls([
            call(expected_download_science_path),
            call(expected_download_ancillary_path_1),
            call(expected_download_ancillary_path_2),
        ])

        # TODO: get rid of test path once the algorithm doc is updated
        test_mass_per_charge_lookup_path = get_test_data_path(os.path.join('codice', 'test_mass_per_charge_lookup.csv'))
        test_esa_lookup_path = get_test_data_path(os.path.join('codice', 'esa_step_lookup.csv'))
        mock_from_file_paths.assert_called_with(expected_download_science_path, test_mass_per_charge_lookup_path,
                                                test_esa_lookup_path)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL2Data.read_from_cdf')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.MassPerChargeLookup.read_from_file')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.ESAStepLookup.read_from_file')
    def test_from_file_paths(self, mock_esa_read_from_file, mock_mpc_read_from_file, mock_read_from_cdf):
        actual_dependencies = CodiceLoL3aDependencies.from_file_paths(sentinel.cdf_path, sentinel.mpc_lookup_path,
                                                                      sentinel.esa_lookup_path)
        mock_mpc_read_from_file.assert_called_once_with(sentinel.mpc_lookup_path)
        mock_esa_read_from_file.assert_called_once_with(sentinel.esa_lookup_path)
        mock_read_from_cdf.assert_called_once_with(sentinel.cdf_path)
        self.assertEqual(mock_read_from_cdf.return_value, actual_dependencies.codice_l2_lo_data)
        self.assertEqual(mock_mpc_read_from_file.return_value, actual_dependencies.mass_per_charge_lookup)
        self.assertEqual(mock_esa_read_from_file.return_value, actual_dependencies.esa_steps_lookup)


if __name__ == '__main__':
    unittest.main()
