import unittest
from pathlib import Path
from unittest.mock import call, patch

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies


class TestCodiceHiL3aDirectEventDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies.download")
    @patch(
        "imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies.CodiceHiL3aDirectEventsDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download):
        input_collection = ProcessingInputCollection()

        expected_codice_l2_direct_event_science_file_download_path = "imap/codice/l2/2010/01/imap_codice_l2_hi-direct-events_20100105_v010.cdf"
        codice_l2_direct_event_input_file_name = "imap_codice_l2_hi-direct-events_20100105_v010.cdf"
        
        expected_codice_l1a_direct_event_science_file_download_path = "imap/codice/l1a/2010/10/imap_codice_l1a_hi-direct-events_20101005_v010.cdf"
        codice_l1a_direct_event_input_file_name = "imap_codice_l1a_hi-direct-events_20101005_v010.cdf"

        expected_codice_hi_mass_correction_lut_ancillary_file_download_path = "codice/imap_codice_l3-hi-mass-correction-lut_20100105_v010.xlsx"
        codice_hi_mass_correction_lut_input_file_name = "imap_codice_l3-hi-mass-correction-lut_20100105_v010.xlsx"

        codice_l2_direct_event_science_input = ScienceInput(codice_l2_direct_event_input_file_name)
        codice_l1a_direct_event_science_input = ScienceInput(codice_l1a_direct_event_input_file_name)
        codice_hi_mass_correction_lut_ancillary_input = AncillaryInput(codice_hi_mass_correction_lut_input_file_name)

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_hi-direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l3_hi-direct-events_20100105_v010.cdf")

        input_collection.add([codice_l1a_direct_event_science_input, codice_l2_direct_event_science_input, codice_hi_mass_correction_lut_ancillary_input, non_codice_ancillary_input, non_codice_science_input,
                              non_l2_codice_science_input])

        codice_l3_dependencies = CodiceHiL3aDirectEventsDependencies.fetch_dependencies(input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_codice_l2_direct_event_science_path = data_dir / expected_codice_l2_direct_event_science_file_download_path
        expected_download_codice_l1a_direct_event_science_path = data_dir / expected_codice_l1a_direct_event_science_file_download_path
        expected_download_codice_hi_mass_correction_lut_ancillary_path = data_dir / "imap/ancillary" / expected_codice_hi_mass_correction_lut_ancillary_file_download_path
        
        
        mock_download.assert_has_calls([
            call(expected_download_codice_l2_direct_event_science_path),
            call(expected_download_codice_l1a_direct_event_science_path),
            call(expected_download_codice_hi_mass_correction_lut_ancillary_path),
        ])

        
        mock_from_file_paths.assert_called_with(
            expected_download_codice_l2_direct_event_science_path,
            expected_download_codice_l1a_direct_event_science_path,
            expected_download_codice_hi_mass_correction_lut_ancillary_path,
        )

        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch("imap_l3_processing.codice.l3.hi.models.CodiceL2HiDirectEventData.read_from_cdf")
    @patch("imap_l3_processing.codice.l3.hi.models.CodiceL1aHiDirectEvents.read_from_cdf")
    @patch("imap_l3_processing.codice.l3.hi.direct_event.codice_hi_mass_correction_lookup_table.CodiceHiMassCorrectionLookupTable.from_file")
    def test_can_load_from_files(self, mock_from_file_mass_correction_lut_ancillary, mock_l1a_read_from_cdf, mock_l2_read_from_cdf):
        codice_l2_cdf_file = Path("CodiceL2CDF")
        codice_l1a_cdf_file = Path("CodiceL1aCDF")
        codice_mass_correction_lut_ancillary_file = Path("CodiceMassCorrectionLutAncillary")

        codice_l3_dependencies = CodiceHiL3aDirectEventsDependencies.from_file_paths(codice_l2_cdf_file, codice_l1a_cdf_file, codice_mass_correction_lut_ancillary_file)

        mock_l2_read_from_cdf.assert_called_with(codice_l2_cdf_file)
        mock_l1a_read_from_cdf.assert_called_with(codice_l1a_cdf_file)
        mock_from_file_mass_correction_lut_ancillary.assert_called_with(codice_mass_correction_lut_ancillary_file)

        self.assertEqual(mock_l2_read_from_cdf.return_value, codice_l3_dependencies.codice_l2_hi_data)
        self.assertEqual(mock_l1a_read_from_cdf.return_value, codice_l3_dependencies.codice_l1a_hi_data)
        self.assertEqual(mock_from_file_mass_correction_lut_ancillary.return_value, codice_l3_dependencies.codice_hi_mass_correction_lut_ancillary)
