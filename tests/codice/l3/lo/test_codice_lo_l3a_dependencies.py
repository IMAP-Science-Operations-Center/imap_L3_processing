import unittest
from unittest.mock import patch, call, sentinel

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies, SW_SPECIES_DESCRIPTOR, \
    MASS_PER_CHARGE_DESCRIPTOR, DIRECT_EVENTS_DESCRIPTOR, PRIORITY_RATES_DESCRIPTOR, MASS_COEFFICIENT_DESCRIPTOR


class TestCodiceLoL3aDependencies(unittest.TestCase):

    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.download_dependency_from_path")
    @patch("imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL3aDependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_utils_download):
        processing_input_collection = ProcessingInputCollection()
        expected_codice_science_file_download_path = f"imap/codice/l2/2010/01/imap_codice_l2_{SW_SPECIES_DESCRIPTOR}_20100105_v010.cdf"
        expected_direct_events_science_file_download_path = f"imap/codice/l2b/2010/01/imap_codice_l2b_{DIRECT_EVENTS_DESCRIPTOR}_20100105_v010.cdf"
        expected_priority_rates_science_file_download_path = f"imap/codice/l2b/2010/01/imap_codice_l2b_{PRIORITY_RATES_DESCRIPTOR}_20100105_v010.cdf"
        expected_mass_per_charge_file_download_path = f"imap/ancillary/codice/imap_codice_{MASS_PER_CHARGE_DESCRIPTOR}_20100105_v001.csv"
        expected_mass_coefficient_file_download_path = f"imap/ancillary/codice/imap_codice_{MASS_COEFFICIENT_DESCRIPTOR}_20100105_v001.csv"

        codice_sectored_intensity_input_file_name = f"imap_codice_l2_{SW_SPECIES_DESCRIPTOR}_20100105_v010.cdf"
        codice_priority_rates_input_file_name = f"imap_codice_l2b_{PRIORITY_RATES_DESCRIPTOR}_20100105_v010.cdf"
        codice_direct_events_input_file_name = f"imap_codice_l2b_{DIRECT_EVENTS_DESCRIPTOR}_20100105_v010.cdf"
        mass_per_charge_lookup_file = f"imap_codice_{MASS_PER_CHARGE_DESCRIPTOR}_20100105_v001.csv"
        mass_coefficient_lookup_file = f"imap_codice_{MASS_COEFFICIENT_DESCRIPTOR}_20100105_v001.csv"

        sectored_science_input = ScienceInput(codice_sectored_intensity_input_file_name)
        ancillary_input_1 = AncillaryInput(mass_per_charge_lookup_file)
        ancillary_input_2 = AncillaryInput(mass_coefficient_lookup_file)
        priority_science_input = ScienceInput(codice_priority_rates_input_file_name)
        direct_events_science_input = ScienceInput(codice_direct_events_input_file_name)

        non_codice_ancillary_input = AncillaryInput(
            "imap/ancillary/hit/imap_codice_range-2A-cosine-lookup_20250203_v001.csv")
        non_codice_science_input = ScienceInput("imap/hit/l3/2010/01/imap_hit_l3_direct-events_20100105_v010.cdf")
        non_l2_codice_science_input = ScienceInput("imap_codice_l1_direct-events_20100105_v010.cdf")

        processing_input_collection.add(
            [sectored_science_input, priority_science_input, direct_events_science_input, ancillary_input_1,
             ancillary_input_2,
             non_codice_ancillary_input, non_codice_science_input,
             non_l2_codice_science_input])

        codice_l3_dependencies = CodiceLoL3aDependencies.fetch_dependencies(processing_input_collection)

        data_dir = imap_data_access.config["DATA_DIR"]
        expected_download_sectored_science_path = data_dir / expected_codice_science_file_download_path
        expected_download_priority_rates_science_path = data_dir / expected_priority_rates_science_file_download_path
        expected_download_direct_events_science_path = data_dir / expected_direct_events_science_file_download_path
        expected_download_ancillary_path_1 = data_dir / expected_mass_per_charge_file_download_path
        expected_download_ancillary_path_2 = data_dir / expected_mass_coefficient_file_download_path

        mock_utils_download.assert_has_calls([
            call(expected_download_sectored_science_path),
            call(expected_download_priority_rates_science_path),
            call(expected_download_direct_events_science_path),
            call(expected_download_ancillary_path_1),
            call(expected_download_ancillary_path_2),
        ], any_order=True)

        mock_from_file_paths.assert_called_with(expected_download_sectored_science_path,
                                                expected_download_priority_rates_science_path,
                                                expected_download_direct_events_science_path,
                                                expected_download_ancillary_path_1,
                                                expected_download_ancillary_path_2)
        self.assertEqual(mock_from_file_paths.return_value, codice_l3_dependencies)

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL2DirectEventData.read_from_cdf')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL2bPriorityRates.read_from_cdf')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.CodiceLoL2SWSpeciesData.read_from_cdf')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.MassPerChargeLookup.read_from_file')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies.MassCoefficientLookup.read_from_csv')
    def test_from_file_paths(self, mock_mass_coefficient_lookup, mock_mpc_read_from_file,
                             mock_read_from_cdf,
                             mock_priority_read_from_cdf, mock_direct_event_read_from_cdf):
        actual_dependencies = CodiceLoL3aDependencies.from_file_paths(sentinel.cdf_path,
                                                                      sentinel.priority_rates_cdf_path,
                                                                      sentinel.direct_events_cdf_path,
                                                                      sentinel.mpc_lookup_path,
                                                                      sentinel.mass_coefficient_lookup_path)

        mock_mpc_read_from_file.assert_called_once_with(sentinel.mpc_lookup_path)
        mock_read_from_cdf.assert_called_once_with(sentinel.cdf_path)
        mock_priority_read_from_cdf.assert_called_once_with(sentinel.priority_rates_cdf_path)
        mock_direct_event_read_from_cdf.assert_called_once_with(sentinel.direct_events_cdf_path)
        mock_mass_coefficient_lookup.assert_called_once_with(sentinel.mass_coefficient_lookup_path)

        self.assertEqual(mock_read_from_cdf.return_value, actual_dependencies.codice_l2_lo_data)
        self.assertEqual(mock_priority_read_from_cdf.return_value, actual_dependencies.codice_l2b_lo_priority_rates)
        self.assertEqual(mock_direct_event_read_from_cdf.return_value, actual_dependencies.codice_l2_direct_events)
        self.assertEqual(mock_mpc_read_from_file.return_value, actual_dependencies.mass_per_charge_lookup)
        self.assertEqual(mock_mass_coefficient_lookup.return_value, actual_dependencies.mass_coefficient_lookup)


if __name__ == '__main__':
    unittest.main()
