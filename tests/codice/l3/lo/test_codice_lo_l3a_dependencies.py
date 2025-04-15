import unittest
from unittest.mock import patch, call

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies


class TestCodiceLoL3aDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.download_dependency_from_path")
    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL3aDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_utils_download):
        processing_input_collection = ProcessingInputCollection()

        expected_codice_science_file_download_path = "imap/codice/l2/2010/01/imap_codice_l2_sectored-intensities_20100105_v010.cdf"
        expected_codice_ancillary_file_download_path = "imap/ancillary/codice/imap_codice_mass-per-charge-lookup_20100105_v001.csv"

        codice_sectored_intensity_input_file_name = "imap_codice_l2_sectored-intensities_20100105_v010.cdf"
        codice_ancillary_file = "imap_codice_mass-per-charge-lookup_20100105_v001.csv"

        science_input = ScienceInput(codice_sectored_intensity_input_file_name)
        ancillary_input = AncillaryInput(codice_ancillary_file)

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        processing_input_collection.add(
            [science_input, ancillary_input, non_codice_ancillary_input, non_codice_science_input,
             non_l2_codice_science_input])

        codice_l3_dependencies = CodiceLoL3aDependencies.fetch_dependencies(processing_input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_science_path = data_dir / expected_codice_science_file_download_path
        expected_download_ancillary_path = data_dir / expected_codice_ancillary_file_download_path
        mock_utils_download.assert_has_calls([
            call(expected_download_science_path),
            call(expected_download_ancillary_path),
        ])

        mock_from_file_paths.assert_called_with(expected_download_science_path, expected_download_ancillary_path)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)


if __name__ == '__main__':
    unittest.main()
