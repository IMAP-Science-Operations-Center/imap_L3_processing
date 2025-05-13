import unittest
from unittest.mock import patch, sentinel, call

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies, MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR
from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import SW_PRIORITY_DESCRIPTOR, \
    NSW_PRIORITY_DESCRIPTOR


class TestCodiceLoL3a3dDistributions(unittest.TestCase):
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.imap_data_access.download")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL3a3dDistributionsDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_3d_distribution_deps_from_file_paths, mock_data_access_download):
        l3a_direct_event_name = "imap_codice_l3a_lo-direct-events_20100105_v010.cdf"
        l1a_sw_priority_name = f"imap_codice_l1a_{SW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        l1a_nsw_priority_name = f"imap_codice_l1a_{NSW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        mass_species_lut_name = f"imap_codice_{MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR}_20100105_v010.cdf"

        unused_science_input = ScienceInput(f"imap_codice_l2_lo-direct-events_20100105_v010.cdf")

        processing_input_collection = ProcessingInputCollection(AncillaryInput(mass_species_lut_name),
                                                                ScienceInput(l3a_direct_event_name),
                                                                ScienceInput(l1a_sw_priority_name),
                                                                ScienceInput(l1a_nsw_priority_name),
                                                                unused_science_input)

        mock_data_access_download.side_effect = [
            sentinel.l3a_direct_event_downloaded_path,
            sentinel.l1a_sw_priority_downloaded_path,
            sentinel.l1a_nsw_priority_downloaded_path,
            sentinel.mass_species_lut_downloaded_path,
        ]

        CodiceLoL3a3dDistributionsDependencies.fetch_dependencies(processing_input_collection)

        mock_data_access_download.assert_has_calls([
            call(l3a_direct_event_name),
            call(l1a_sw_priority_name),
            call(l1a_nsw_priority_name),
            call(mass_species_lut_name),
        ])

        mock_3d_distribution_deps_from_file_paths.assert_called_once_with(
            l3a_file_path=sentinel.l3a_direct_event_downloaded_path,
            l1a_sw_file_path=sentinel.l1a_sw_priority_downloaded_path,
            l1a_nsw_file_path=sentinel.l1a_nsw_priority_downloaded_path,
            mass_species_bin_lut=sentinel.mass_species_lut_downloaded_path,
        )

    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.MassSpeciesBinLookup.read_from_csv')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL1aSWPriorityRates.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoL1aNSWPriorityRates.read_from_cdf')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies.CodiceLoDirectEventData.read_from_cdf')
    def test_from_file_paths(self, mock_l3a_direct_event_read_from_cdf, mock_l1a_nsw_read_from_cdf,
                             mock_l1a_sw_read_from_cdf, mock_mass_species_from_csv):
        actual = CodiceLoL3a3dDistributionsDependencies.from_file_paths(sentinel.l3a_path, sentinel.l1a_sw_path,
                                                                        sentinel.l1a_nsw_path,
                                                                        sentinel.mass_species_path)
        mock_l3a_direct_event_read_from_cdf.assert_called_once_with(sentinel.l3a_path)
        mock_l1a_nsw_read_from_cdf.assert_called_once_with(sentinel.l1a_nsw_path)
        mock_l1a_sw_read_from_cdf.assert_called_once_with(sentinel.l1a_sw_path)
        mock_mass_species_from_csv.assert_called_once_with(sentinel.mass_species_path)

        expected_dependencies = CodiceLoL3a3dDistributionsDependencies(mock_l3a_direct_event_read_from_cdf.return_value,
                                                                       mock_l1a_sw_read_from_cdf.return_value,
                                                                       mock_l1a_nsw_read_from_cdf.return_value,
                                                                       mock_mass_species_from_csv.return_value)

        self.assertEqual(expected_dependencies, actual)
