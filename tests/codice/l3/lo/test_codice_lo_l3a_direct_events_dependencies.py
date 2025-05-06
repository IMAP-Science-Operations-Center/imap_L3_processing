import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import \
    CodiceLoL3aDirectEventsDependencies, DIRECT_EVENTS_DESCRIPTOR, SW_PRIORITY_DESCRIPTOR, NSW_PRIORITY_DESCRIPTOR, \
    MASS_COEFFICIENT_DESCRIPTOR


class TestCodiceLoL3aDirectEventsDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.download")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.CodiceLoL3aDirectEventsDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download):
        expected_direct_events_science_file_download_path = f"imap/codice/l2b/2010/01/imap_codice_l2_{DIRECT_EVENTS_DESCRIPTOR}_20100105_v010.cdf"
        expected_lo_sw_priority_science_file_download_path = f"imap/codice/l1a/2010/01/imap_codice_l1a_{SW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        expected_lo_nsw_priority_science_file_download_path = f"imap/codice/l1a/2010/01/imap_codice_l1a_{NSW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        expected_mass_coefficient_file_download_path = f"imap/ancillary/codice/imap_codice_{MASS_COEFFICIENT_DESCRIPTOR}_20100105_v001.csv"

        mock_download.side_effect = [
            expected_lo_sw_priority_science_file_download_path,
            expected_lo_nsw_priority_science_file_download_path,
            expected_direct_events_science_file_download_path,
            expected_mass_coefficient_file_download_path
        ]

        ancillary_input = AncillaryInput(f"imap_codice_{MASS_COEFFICIENT_DESCRIPTOR}_20100105_v001.csv")
        sw_priority_science_input = ScienceInput(f"imap_codice_l1a_{SW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf")
        nsw_priority_science_input = ScienceInput(f"imap_codice_l1a_{NSW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf")
        direct_events_science_input = ScienceInput(f"imap_codice_l2_{DIRECT_EVENTS_DESCRIPTOR}_20100105_v010.cdf")

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        processing_input_collection = ProcessingInputCollection(
            sw_priority_science_input,
            nsw_priority_science_input,
            direct_events_science_input,
            ancillary_input,
            non_codice_ancillary_input,
            non_codice_science_input,
            non_l2_codice_science_input)

        codice_l3_dependencies = CodiceLoL3aDirectEventsDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(Path(expected_lo_sw_priority_science_file_download_path).name),
            call(Path(expected_lo_nsw_priority_science_file_download_path).name),
            call(Path(expected_direct_events_science_file_download_path).name),
            call(Path(expected_mass_coefficient_file_download_path).name),
        ], any_order=True)

        mock_from_file_paths.assert_called_with(expected_lo_sw_priority_science_file_download_path,
                                                expected_lo_nsw_priority_science_file_download_path,
                                                expected_direct_events_science_file_download_path,
                                                expected_mass_coefficient_file_download_path)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.CodiceLoL2DirectEventData.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.MassCoefficientLookup.read_from_csv')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.CodiceLoL1aSWPriorityRates.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.CodiceLoL1aNSWPriorityRates.read_from_cdf')
    def test_from_file_paths(self, mock_codice_lo_l1a_nsw_priority_rates,
                             mock_codice_lo_l1a_sw_priority_rates,
                             mock_mass_coefficient_lookup,
                             mock_direct_event_read_from_cdf):
        actual_dependencies = CodiceLoL3aDirectEventsDependencies.from_file_paths(
            sentinel.sw_priority,
            sentinel.nsw_priority,
            sentinel.direct_events_cdf_path,
            sentinel.mass_coefficient_lookup_path)

        mock_direct_event_read_from_cdf.assert_called_once_with(sentinel.direct_events_cdf_path)
        mock_mass_coefficient_lookup.assert_called_once_with(sentinel.mass_coefficient_lookup_path)
        mock_codice_lo_l1a_sw_priority_rates.assert_called_once_with(sentinel.sw_priority)
        mock_codice_lo_l1a_nsw_priority_rates.assert_called_once_with(sentinel.nsw_priority)

        self.assertEqual(mock_direct_event_read_from_cdf.return_value, actual_dependencies.codice_l2_direct_events)
        self.assertEqual(mock_mass_coefficient_lookup.return_value, actual_dependencies.mass_coefficient_lookup)
        self.assertEqual(mock_codice_lo_l1a_sw_priority_rates.return_value,
                         actual_dependencies.codice_lo_l1a_sw_priority_rates)
        self.assertEqual(mock_codice_lo_l1a_nsw_priority_rates.return_value,
                         actual_dependencies.codice_lo_l1a_nsw_priority_rates)


if __name__ == '__main__':
    unittest.main()
