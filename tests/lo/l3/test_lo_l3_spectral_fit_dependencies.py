import unittest
from unittest.mock import patch

from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection

from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies


class TestLoL3SpectralFitDependencies(unittest.TestCase):

    @patch("imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies.RectangularIntensityMapData.read_from_path")
    @patch("imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies.download")
    def test_fetch_dependencies(self, mock_download, mock_read_from_path):
        for input_data_level in ("l2", "l3"):
            mock_download.reset_mock()
            with self.subTest(input_data_level):
                file_name = f"imap_lo_{input_data_level}_l90-ena-h-hf-sp-full-hae-4deg-6mo_20250422_v001.cdf"
                pointing_set = "imap_lo_l1c_pset_20260101-repoint01261_v001.cdf"
                glows_file = "imap_glows_l3e_survival-repoint00010_20250422_v001.cdf"
                ena_input = ScienceInput(file_name)
                extra_inputs = [ScienceInput(pointing_set), ScienceInput(glows_file)]

                input_deps = ProcessingInputCollection(ena_input, *extra_inputs)

                dependencies = LoL3SpectralFitDependencies.fetch_dependencies(input_deps)

                mock_download.assert_called_once_with(file_name)
                self.assertEqual(dependencies.map_data, mock_read_from_path.return_value)

                mock_read_from_path.assert_called_with(mock_download.return_value)

    def test_raises_error_if_unexpected_number_of_files(self):
        l3_file_name = "imap_lo_l3_l090-ena-h-hf-sp-ram-hae-4deg-6mo_20250422_v001.cdf"
        l2_file_name = "imap_lo_l2_l090-ena-h-hf-nsp-ram-hae-4deg-6mo_20250422_v001.cdf"
        cases = [
            ("0 files", ProcessingInputCollection()),
            ("2 L3 files", ProcessingInputCollection(
                ScienceInput(l3_file_name),
                ScienceInput(l3_file_name),
            )),
             ("1 L2 and 1 L3 file", ProcessingInputCollection(ScienceInput(l2_file_name), ScienceInput(l3_file_name))),
             ("2 L2 files", ProcessingInputCollection(ScienceInput(l2_file_name), ScienceInput(l2_file_name)))
        ]

        for name, collection in cases:
            with self.subTest(name):
                with self.assertRaises(ValueError) as e:
                    LoL3SpectralFitDependencies.fetch_dependencies(collection)

                self.assertEqual("Incorrect number of dependencies", str(e.exception))
