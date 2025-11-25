import unittest
from pathlib import Path
from unittest.mock import patch, Mock, sentinel, call

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.models import Instrument


class TestHiLoL3SurvivalDependencies(unittest.TestCase):

    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.parse_map_descriptor")
    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.imap_data_access.download")
    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.read_glows_l3e_data")
    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.read_l1c_rectangular_pointing_set_data")
    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.RectangularIntensityMapData.read_from_path")
    def test_fetch_dependencies(self, mock_read_from_path, mock_read_l1c, mock_read_glows_l3e,
                                mock_imap_data_access_download, mock_parse_map_descriptor):
        for instrument in [Instrument.IMAP_HI, Instrument.IMAP_LO]:
            mock_read_from_path.reset_mock()
            mock_parse_map_descriptor.reset_mock()

            l1c_file_names = [f"imap_{instrument.value}_l1c_45sensor-pset_20201001_v001.cdf",
                              f"imap_{instrument.value}_l1c_45sensor-pset_20201002_v002.cdf",
                              f"imap_{instrument.value}_l1c_45sensor-pset_20201003_v001.cdf"]
            glows_file_names = [
                f"imap_glows_l3e_survival-probability-{instrument.value}_20201001_v001.cdf",
                f"imap_glows_l3e_survival-probability-{instrument.value}_20201002_v002.cdf",
                f"imap_glows_l3e_survival-probability-{instrument.value}_20201003_v003.cdf"]

            l2_map_descriptor = f"ena-h-sf-nsp-ram-hae-6deg-1yr"
            l2_filename = f"imap_{instrument.value}_l2_{l2_map_descriptor}_20250415_v001.cdf"

            dependencies = ProcessingInputCollection(
                ScienceInput(l2_filename),
                *[ScienceInput(l1c) for l1c in l1c_file_names],
                *[ScienceInput(l3e) for l3e in glows_file_names]
            )

            downloaded_files = [sentinel.l2_file_path, sentinel.l1c_file_path_1,
                                sentinel.l1c_file_path_2,
                                sentinel.l1c_file_path_3, sentinel.glows_file_path_1,
                                sentinel.glows_file_path_2, sentinel.glows_file_path_3]

            mock_imap_data_access_download.side_effect = downloaded_files
            mock_read_l1c.side_effect = [sentinel.l1c_data_1, sentinel.l1c_data_2, sentinel.l1c_data_3]

            mock_read_glows_l3e.side_effect = [sentinel.glows_data_1, sentinel.glows_data_2,
                                               sentinel.glows_data_3]

            actual = HiLoL3SurvivalDependencies.fetch_dependencies(dependencies, instrument)

            expected_imap_data_access_calls = [call(l2_filename)] + [call(path) for path in
                                                                     l1c_file_names + glows_file_names]
            mock_imap_data_access_download.assert_has_calls(expected_imap_data_access_calls)

            mock_read_from_path.assert_called_once_with(sentinel.l2_file_path)

            mock_read_l1c.assert_has_calls([
                call(sentinel.l1c_file_path_1),
                call(sentinel.l1c_file_path_2),
                call(sentinel.l1c_file_path_3)
            ])

            mock_read_glows_l3e.assert_has_calls([
                call(sentinel.glows_file_path_1),
                call(sentinel.glows_file_path_2),
                call(sentinel.glows_file_path_3)
            ])

            mock_parse_map_descriptor.assert_called_once_with(l2_map_descriptor)

            self.assertEqual(actual.l2_data, mock_read_from_path.return_value)
            self.assertEqual(actual.l1c_data, [sentinel.l1c_data_1,
                                               sentinel.l1c_data_2,
                                               sentinel.l1c_data_3])
            self.assertEqual(actual.glows_l3e_data, [sentinel.glows_data_1,
                                                     sentinel.glows_data_2,
                                                     sentinel.glows_data_3, ])
            self.assertEqual(actual.l2_map_descriptor_parts, mock_parse_map_descriptor.return_value)
            self.assertEqual(actual.dependency_file_paths, downloaded_files)

    @patch("imap_l3_processing.maps.hilo_l3_survival_dependencies.HiLoL3SurvivalDependencies.fetch_dependencies")
    def test_fetch_single_sensor_full_spin_dependencies(self, mock_fetch_dependencies):
        cases = [("sf", False), ("hf", True)]
        for reference_frame, expected_cg_corrected in cases:
            mock_fetch_dependencies.reset_mock()
            with self.subTest(reference_frame=reference_frame):
                ram_map_descriptor = f"h90-ena-h-{reference_frame}-nsp-ram-hae-6deg-1yr"
                ram_l2_filename = f"imap_hi_l2_{ram_map_descriptor}_20250415_v001.cdf"

                anti_map_descriptor = f"h90-ena-h-{reference_frame}-nsp-anti-hae-6deg-1yr"
                anti_l2_filename = f"imap_hi_l2_{anti_map_descriptor}_20250415_v001.cdf"
                glows_l3e_filename = "imap_glows_l3e_survival-probability_20250415_v001.cdf"
                l1c_filename = "imap_hi_l1c_pset_20250415-repoint10000_v001.cdf"

                ram_input = ScienceInput(ram_l2_filename)
                anti_input = ScienceInput(anti_l2_filename)
                glows_input = ScienceInput(glows_l3e_filename)
                l1c_input = ScienceInput(l1c_filename)
                dependencies = ProcessingInputCollection(ram_input, anti_input, glows_input, l1c_input)

                mock_ram_survival_dependencies = Mock(spec=HiLoL3SurvivalDependencies)
                mock_ram_survival_dependencies.dependency_file_paths = [Path("ram_input1"), Path("ram_input2")]
                mock_antiram_survival_dependencies = Mock(spec=HiLoL3SurvivalDependencies)
                mock_antiram_survival_dependencies.dependency_file_paths = [Path("antiram_input1"),
                                                                            Path("antiram_input2")]

                mock_fetch_dependencies.side_effect = [mock_ram_survival_dependencies,
                                                       mock_antiram_survival_dependencies]

                result = HiL3SingleSensorFullSpinDependencies.fetch_dependencies(dependencies)
                self.assertIsInstance(result, HiL3SingleSensorFullSpinDependencies)
                self.assertEqual(result.ram_dependencies, mock_ram_survival_dependencies)
                self.assertEqual(result.antiram_dependencies, mock_antiram_survival_dependencies)
                self.assertEqual(expected_cg_corrected, result.cg_corrected)

                [ram_dependencies, instrument] = mock_fetch_dependencies.call_args_list[0].args
                self.assertIsInstance(ram_dependencies, ProcessingInputCollection)
                self.assertEqual([ram_input, glows_input, l1c_input], ram_dependencies.processing_input)
                self.assertEqual(Instrument.IMAP_HI, instrument)

                [antiram_dependencies, instrument] = mock_fetch_dependencies.call_args_list[1].args
                self.assertIsInstance(antiram_dependencies, ProcessingInputCollection)
                self.assertEqual([anti_input, glows_input, l1c_input], antiram_dependencies.processing_input)
                self.assertEqual(Instrument.IMAP_HI, instrument)

                self.assertEqual(
                    [Path("ram_input1"), Path("ram_input2"), Path("antiram_input1"), Path("antiram_input2")],
                    result.dependency_file_paths)
