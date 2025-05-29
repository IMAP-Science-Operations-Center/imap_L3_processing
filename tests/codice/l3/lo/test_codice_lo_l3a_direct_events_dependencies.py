import unittest
from pathlib import Path
from unittest.mock import patch, call, sentinel

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import \
    CodiceLoL3aDirectEventsDependencies, DIRECT_EVENTS_DESCRIPTOR, SW_PRIORITY_DESCRIPTOR, NSW_PRIORITY_DESCRIPTOR, \
    MASS_COEFFICIENT_DESCRIPTOR
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import \
    ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR


class TestCodiceLoL3aDirectEventsDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.download")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.CodiceLoL3aDirectEventsDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download):
        direct_events_science_file_name = f"imap_codice_l2_{DIRECT_EVENTS_DESCRIPTOR}_20100105_v010.cdf"
        lo_sw_priority_science_file_name = f"imap_codice_l1a_{SW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        lo_nsw_priority_science_file_name = f"imap_codice_l1a_{NSW_PRIORITY_DESCRIPTOR}_20100105_v010.cdf"
        mass_coefficient_file_name = f"imap_codice_{MASS_COEFFICIENT_DESCRIPTOR}_20100105_v001.csv"
        energy_lookup_file_name = f"imap_codice_{ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR}_20100105_v001.csv"

        mock_download.side_effect = [
            Path("some_local_folder") / lo_sw_priority_science_file_name,
            Path("some_local_folder") / lo_nsw_priority_science_file_name,
            Path("some_local_folder") / direct_events_science_file_name,
            Path("some_local_folder") / mass_coefficient_file_name,
            Path("some_local_folder") / energy_lookup_file_name
        ]

        processing_input_collection = ProcessingInputCollection(
            AncillaryInput(energy_lookup_file_name),
            AncillaryInput(mass_coefficient_file_name),
            ScienceInput(lo_nsw_priority_science_file_name),
            ScienceInput(lo_sw_priority_science_file_name),
            ScienceInput(direct_events_science_file_name),

            AncillaryInput("imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv"),
            ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf"),
            ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")
        )

        codice_l3_dependencies = CodiceLoL3aDirectEventsDependencies.fetch_dependencies(processing_input_collection)

        mock_download.assert_has_calls([
            call(lo_sw_priority_science_file_name),
            call(lo_nsw_priority_science_file_name),
            call(direct_events_science_file_name),
            call(mass_coefficient_file_name),
            call(energy_lookup_file_name)
        ], any_order=True)

        mock_from_file_paths.assert_called_with(
            Path("some_local_folder") / lo_sw_priority_science_file_name,
            Path("some_local_folder") / lo_nsw_priority_science_file_name,
            Path("some_local_folder") / direct_events_science_file_name,
            Path("some_local_folder") / mass_coefficient_file_name,
            Path("some_local_folder") / energy_lookup_file_name
        )
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies.EnergyLookup.read_from_csv')
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
                             mock_direct_event_read_from_cdf, mock_energy_lookup_read_from_csv):
        actual_dependencies = CodiceLoL3aDirectEventsDependencies.from_file_paths(
            sentinel.sw_priority,
            sentinel.nsw_priority,
            sentinel.direct_events_cdf_path,
            sentinel.mass_coefficient_lookup_path,
            sentinel.energy_lookup_path
        )

        mock_direct_event_read_from_cdf.assert_called_once_with(sentinel.direct_events_cdf_path)
        mock_mass_coefficient_lookup.assert_called_once_with(sentinel.mass_coefficient_lookup_path)
        mock_codice_lo_l1a_sw_priority_rates.assert_called_once_with(sentinel.sw_priority)
        mock_codice_lo_l1a_nsw_priority_rates.assert_called_once_with(sentinel.nsw_priority)
        mock_energy_lookup_read_from_csv.assert_called_once_with(sentinel.energy_lookup_path)

        self.assertEqual(mock_direct_event_read_from_cdf.return_value, actual_dependencies.codice_l2_direct_events)
        self.assertEqual(mock_mass_coefficient_lookup.return_value, actual_dependencies.mass_coefficient_lookup)
        self.assertEqual(mock_codice_lo_l1a_sw_priority_rates.return_value,
                         actual_dependencies.codice_lo_l1a_sw_priority_rates)
        self.assertEqual(mock_codice_lo_l1a_nsw_priority_rates.return_value,
                         actual_dependencies.codice_lo_l1a_nsw_priority_rates)
        self.assertEqual(mock_energy_lookup_read_from_csv.return_value, actual_dependencies.energy_lookup)


if __name__ == '__main__':
    unittest.main()
