import unittest
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies


class TestUltraL3Dependencies(unittest.TestCase):
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.download')
    def test_fetch_dependencies(self, mock_from_file_path, mock_read_ultra, mock_read_glows):
        ultra_input = ScienceInput('imap_ultra_l1c_pset_20250101_v001.cdf')
        glows_input = ScienceInput('imap_glows_l3e_survival-probability_20250101_v001.cdf')
        mock_read_ultra.return_value = sentinel.ultra_data
        mock_read_glows.return_value = sentinel.glows_data

        input_collection = ProcessingInputCollection(ultra_input, glows_input)

        mock_from_file_path.assert_has_calls([call(file_path) for file_path in input_collection.get_file_paths()])

        dependencies = UltraL3Dependencies.fetch_dependencies(input_collection)
        self.assertEqual(sentinel.ultra_data, dependencies.ultra_l1c_pset)
        self.assertEqual(sentinel.glows_data, dependencies.glows_l3e_sp)

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    def test_from_file_paths(self, mock_read_ultra: Mock, mock_read_glows: Mock):
        ultra_input_paths = [Path("u_path_1"), Path("u_path_2")]
        glows_input_paths = [Path("g_path_1"), Path("g_path_2")]

        mock_read_ultra.side_effect = [sentinel.ultra_data1, sentinel.ultra_data2]
        mock_read_glows.side_effect = [sentinel.glows_data1, sentinel.glows_data2]

        result = UltraL3Dependencies.read_from_file(ultra_input_paths, glows_input_paths)

        mock_read_ultra.assert_has_calls([call(file_path) for file_path in ultra_input_paths])
        mock_read_glows.assert_has_calls([call(file_path) for file_path in glows_input_paths])

        self.assertEqual(result.ultra_l1c_pset, [sentinel.ultra_data1, sentinel.ultra_data2])
        self.assertEqual(result.glows_l3e_sp, [sentinel.glows_data1, sentinel.glows_data2])
