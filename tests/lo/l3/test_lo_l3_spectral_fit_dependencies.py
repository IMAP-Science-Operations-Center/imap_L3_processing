import unittest
from unittest.mock import patch

from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies


class TestLoL3SpectralFitDependencies(unittest.TestCase):

    @patch("imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies.RectangularIntensityMapData.read_from_path")
    @patch("imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_read_from_path):
        file_name = "imap_lo_l3_l90-ena-h-hf-sp-full-hae-4deg-6mo_20250422_v001.cdf"
        input = ScienceInput(file_name)
        dependencies = ProcessingInputCollection(input)

        dependencies = LoL3SpectralFitDependencies.fetch_dependencies(dependencies)

        mock_download.assert_called_once_with(file_name)
        self.assertEqual(dependencies.map_data, mock_read_from_path.return_value)

        mock_read_from_path.assert_called_with(mock_download.return_value)

    def test_raises_error_if_unexpected_number_ef_files(self):
        file_name = "imap_lo_l3_l90-ena-h-hf-sp-full-hae-4deg-6mo_20250422_v001.cdf"
        cases = [
            ("0 files", ProcessingInputCollection()),
            ("2 files", ProcessingInputCollection(
                ScienceInput(file_name),
                ScienceInput(file_name),
            ))
        ]

        for name, collection in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError) as e:
                    LoL3SpectralFitDependencies.fetch_dependencies(collection)

                self.assertEqual("Incorrect number of dependencies", str(e.exception))
