import unittest
from unittest.mock import patch, sentinel

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_3d_distributions_dependencies import \
    CodiceLoL3a3dDistributionsDependencies


class TestCodiceLoL3a3dDistributions(unittest.TestCase):
    def test_fetch_dependencies(self):
        pass

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
