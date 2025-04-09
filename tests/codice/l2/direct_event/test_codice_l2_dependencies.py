import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies import CodiceL2Dependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestCodiceL2Dependencies(unittest.TestCase):
    @patch("imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies.download_dependency")
    @patch("imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies.CodiceL2Dependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_files, mock_download_dependency):
        codice_l1a_direct_events = UpstreamDataDependency(instrument="codice", data_level="l1a", start_date=None,
                                                          end_date=None, descriptor="direct-events", version=None)

        mock_download_dependency.side_effect = ["l1a_codice_cdf_file",
                                                "energy_lookup_file_path",
                                                "energy_bin_lookup_file_path",
                                                "tof_lookup_file_path",
                                                "azimuth_lookup_file_path"]

        expected_codice_dependencies = CodiceL2Dependencies(codice_l1a_hi_data=Mock(), energy_lookup_table=Mock(),
                                                            time_of_flight_lookup_table=Mock(),
                                                            azimuth_lookup_table=Mock())
        mock_from_files.return_value = expected_codice_dependencies

        codice_dependencies = CodiceL2Dependencies.fetch_dependencies([codice_l1a_direct_events])

        codice_l1a_upstream_data_dependency = mock_download_dependency.call_args_list[0].args[0]
        energy_lookup_upstream_data_dependency = mock_download_dependency.call_args_list[1].args[0]
        energy_bin_lookup_upstream_data_dependency = mock_download_dependency.call_args_list[2].args[0]
        tof_lookup_upstream_data_dependency = mock_download_dependency.call_args_list[3].args[0]
        azimuth_lookup_upstream_data_dependency = mock_download_dependency.call_args_list[4].args[0]

        self.assertEqual(codice_l1a_upstream_data_dependency, codice_l1a_direct_events)

        self.assertEqual(energy_lookup_upstream_data_dependency.instrument, "codice")
        self.assertEqual(energy_lookup_upstream_data_dependency.data_level, "l2")
        self.assertEqual(energy_lookup_upstream_data_dependency.descriptor, "energy-lookup")

        self.assertEqual(energy_bin_lookup_upstream_data_dependency.instrument, "codice")
        self.assertEqual(energy_bin_lookup_upstream_data_dependency.data_level, "l2")
        self.assertEqual(energy_bin_lookup_upstream_data_dependency.descriptor, "energy-bin-lookup")

        self.assertEqual(tof_lookup_upstream_data_dependency.instrument, "codice")
        self.assertEqual(tof_lookup_upstream_data_dependency.data_level, "l2")
        self.assertEqual(tof_lookup_upstream_data_dependency.descriptor, "time-of-flight-lookup")

        self.assertEqual(azimuth_lookup_upstream_data_dependency.instrument, "codice")
        self.assertEqual(azimuth_lookup_upstream_data_dependency.data_level, "l2")
        self.assertEqual(azimuth_lookup_upstream_data_dependency.descriptor, "azimuth-lookup")

        mock_from_files.assert_called_with("l1a_codice_cdf_file",
                                           "energy_lookup_file_path",
                                           "energy_bin_lookup_file_path",
                                           "tof_lookup_file_path",
                                           "azimuth_lookup_file_path")

        self.assertEqual(codice_dependencies.energy_lookup_table, expected_codice_dependencies.energy_lookup_table)
        self.assertEqual(codice_dependencies.codice_l1a_hi_data, expected_codice_dependencies.codice_l1a_hi_data)
        self.assertEqual(codice_dependencies.azimuth_lookup_table, expected_codice_dependencies.azimuth_lookup_table)
        self.assertEqual(codice_dependencies.time_of_flight_lookup_table,
                         expected_codice_dependencies.time_of_flight_lookup_table)

    @patch("imap_l3_processing.codice.l2.direct_event.science.energy_lookup.EnergyLookup.from_files")
    @patch("imap_l3_processing.codice.l2.direct_event.science.time_of_flight_lookup.TimeOfFlightLookup.from_files")
    @patch("imap_l3_processing.codice.l2.direct_event.science.azimuth_lookup.AzimuthLookup.from_files")
    @patch("imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies.CodiceL2HiData.read_from_cdf")
    def test_can_load_from_files(self, mock_read_from_cdf, mock_azimuth_lookup_from_file, mock_tof_lookup_from_file,
                                 mock_energy_lookup_from_files):
        energy_lookup_file = Path("energy")
        codice_l1_cdf_file = Path("l1 cdf")
        energy_bin_lookup_file = Path("energy_bin")
        tof_lookup_file = Path("tof")
        azimuth_lookup_file = Path("azimuth")

        codice_l2_dependencies = CodiceL2Dependencies.from_file_paths(codice_l1_cdf_file, energy_lookup_file,
                                                                      energy_bin_lookup_file,
                                                                      tof_lookup_file,
                                                                      azimuth_lookup_file)
        mock_read_from_cdf.assert_called_with(codice_l1_cdf_file)
        mock_azimuth_lookup_from_file.assert_called_with(azimuth_lookup_file)
        mock_tof_lookup_from_file.assert_called_with(tof_lookup_file)
        mock_energy_lookup_from_files.assert_called_with(energy_lookup_file, energy_bin_lookup_file)

        self.assertEqual(mock_read_from_cdf.return_value, codice_l2_dependencies.codice_l1a_hi_data)
        self.assertEqual(mock_azimuth_lookup_from_file.return_value, codice_l2_dependencies.azimuth_lookup_table)
        self.assertEqual(mock_tof_lookup_from_file.return_value, codice_l2_dependencies.time_of_flight_lookup_table)
        self.assertEqual(mock_energy_lookup_from_files.return_value, codice_l2_dependencies.energy_lookup_table)
