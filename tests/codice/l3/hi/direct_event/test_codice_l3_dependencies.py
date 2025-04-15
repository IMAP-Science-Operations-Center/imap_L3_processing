import unittest
from pathlib import Path
from unittest.mock import patch, call

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.hi.direct_event.codice_l3_dependencies import CodiceL3Dependencies


class TestCodiceL3Dependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.hi.direct_event.codice_l3_dependencies.download")
    @patch("imap_l3_processing.codice.l3.hi.direct_event.codice_l3_dependencies.CodiceL3Dependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download):
        input_collection = ProcessingInputCollection()

        expected_codice_science_file_download_path = "imap/codice/l2/2010/01/imap_codice_l2_direct-events_20100105_v010.cdf"
        codice_direct_event_input_file_name = "imap_codice_l2_direct-events_20100105_v010.cdf"

        expected_codice_ancillary_file_download_path = "imap/ancillary/codice/imap_codice_tof-lookup_20100105_v001.csv"
        codice_ancillary_file = "imap_codice_tof-lookup_20100105_v001.csv"

        science_input = ScienceInput(codice_direct_event_input_file_name)
        ancillary_input = AncillaryInput(codice_ancillary_file)

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        input_collection.add([science_input, ancillary_input, non_codice_ancillary_input, non_codice_science_input,
                              non_l2_codice_science_input])

        codice_l3_dependencies = CodiceL3Dependencies.fetch_dependencies(input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_science_path = data_dir / expected_codice_science_file_download_path
        expected_download_ancillary_path = data_dir / expected_codice_ancillary_file_download_path
        mock_download.assert_has_calls([
            call(expected_download_science_path),
            call(expected_download_ancillary_path),
        ])

        mock_from_file_paths.assert_called_with(expected_download_science_path, expected_download_ancillary_path)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch("imap_l3_processing.codice.l3.hi.models.CodiceL2HiData.read_from_cdf")
    @patch("imap_l3_processing.codice.l3.hi.direct_event.science.tof_lookup.TOFLookup.read_from_file")
    def test_can_load_from_files(self, mock_tof_lookup_from_file, mock_read_cdf):
        tof_lookup_file = Path("energy_per_nuc")
        codice_l2_cdf_file = Path("CodiceL2CDF")

        codice_l3_dependencies = CodiceL3Dependencies.from_file_paths(codice_l2_cdf_file, tof_lookup_file)

        mock_tof_lookup_from_file.assert_called_with(tof_lookup_file)
        mock_read_cdf.assert_called_with(codice_l2_cdf_file)

        self.assertEqual(mock_tof_lookup_from_file.return_value, codice_l3_dependencies.tof_lookup)
        self.assertEqual(mock_read_cdf.return_value, codice_l3_dependencies.codice_l2_hi_data)
