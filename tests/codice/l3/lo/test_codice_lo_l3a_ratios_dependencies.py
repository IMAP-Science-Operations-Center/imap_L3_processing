import unittest
from unittest.mock import patch, sentinel

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies import CodiceLoL3aRatiosDependencies


class TestCodiceLoL3aRatiosDependencies(unittest.TestCase):

    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies.CodiceLoPartialDensityData.read_from_cdf')
    def test_from_file_paths(self, mock_read_from_cdf):
        actual_dependencies = CodiceLoL3aRatiosDependencies.from_file_paths(sentinel.cdf_path)

        mock_read_from_cdf.assert_called_once_with(sentinel.cdf_path)

        self.assertEqual(mock_read_from_cdf.return_value, actual_dependencies.partial_density_data)

    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies.CodiceLoPartialDensityData.read_from_cdf")
    @patch(
        "imap_l3_processing.codice.l3.lo.codice_lo_l3a_ratios_dependencies.imap_data_access.download")
    def test_fetch_dependencies(self, mock_download, mock_read_from_cdf):
        expected_file_name = "imap_codice_l3a_lo-partial-densities_20200507_v002.cdf"
        science_input = ScienceInput(expected_file_name)
        processing_inputs = ProcessingInputCollection(science_input)

        dependencies = CodiceLoL3aRatiosDependencies.fetch_dependencies(processing_inputs)

        mock_download.assert_called_once_with(expected_file_name)
        self.assertEqual(mock_read_from_cdf.return_value, dependencies.partial_density_data)
