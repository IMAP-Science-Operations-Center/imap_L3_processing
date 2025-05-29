import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock

from spacepy.pycdf import CDF

from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies


class TestUltraL3Dependencies(unittest.TestCase):
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.find_glows_l3e_dependencies')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.download')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.ultra_l2')
    def test_fetch_dependencies(self, mock_ultra_l2, mock_download, mock_read_xarray, mock_read_ultra_l1c,
                                mock_read_glows,
                                mock_find_glows):
        l1c_input_paths = ["imap_ultra_l1c_pset_20251010_v001.cdf", "imap_ultra_l1c_pset_20251011_v001.cdf",
                           "imap_ultra_l1c_pset_20251012_v001.cdf"]

        parents = l1c_input_paths + ["imap_hi_ancil-settings-dontuse_20100101_v001.json"]

        glows_file_paths = [
            "imap_glows_l3e_survival-probability-ultra-45_20201001_v001.cdf",
            "imap_glows_l3e_survival-probability-ultra-45_20201002_v002.cdf",
            "imap_glows_l3e_survival-probability-ultra-45_20201003_v003.cdf"]

        with tempfile.TemporaryDirectory() as tmpdir:
            l2_map_path = Path(tmpdir) / "l2_map.cdf"
            with CDF(str(l2_map_path), masterpath='') as l2_map:
                l2_map.attrs["Parents"] = parents

            input_collection = Mock()
            input_collection.get_file_paths.return_value = [sentinel.imap_l2_map_path]

            returned_download_paths = [l2_map_path, sentinel.l1c_path_1, sentinel.l1c_path_2, sentinel.l1c_path_3,
                                       sentinel.glows_path_1, sentinel.glows_path_2, sentinel.glows_path_3]

            mock_download.side_effect = returned_download_paths
            mock_find_glows.return_value = glows_file_paths
            mock_read_xarray.return_value = sentinel.ultra_l2_data

            l1c_data = [sentinel.ultra_l1c_data_1, sentinel.ultra_l1c_data_2,
                        sentinel.ultra_l1c_data_3]
            mock_read_ultra_l1c.side_effect = l1c_data

            glows_l3e_data = [sentinel.glows_data_1, sentinel.glows_data_2, sentinel.glows_data_3]
            mock_read_glows.side_effect = glows_l3e_data
            mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map]

            dependencies = UltraL3Dependencies.fetch_dependencies(input_collection)

            input_collection.get_file_paths.assert_called_once_with("ultra")
            expected_data_dictionary = {"l1c_path_1": sentinel.l1c_path_1,
                                        "l1c_path_2": sentinel.l1c_path_2,
                                        "l1c_path_3": sentinel.l1c_path_3, }
            mock_ultra_l2.assert_has_calls([call(expected_data_dictionary)])

            mock_find_glows.assert_called_with(l1c_input_paths, "ultra")
            expected_parent_file_paths = [sentinel.imap_l2_map_path, *l1c_input_paths, *glows_file_paths]
            mock_download.assert_has_calls([call(file_path) for file_path in
                                            expected_parent_file_paths])
            mock_read_xarray.assert_called_once_with(sentinel.ultra_l2_healpix_map)

            self.assertEqual(l1c_data, dependencies.ultra_l1c_pset)
            self.assertEqual(glows_l3e_data, dependencies.glows_l3e_sp)
            self.assertEqual(sentinel.ultra_l2_data, dependencies.ultra_l2_map)
            self.assertEqual(returned_download_paths, dependencies.dependency_file_paths)

    def test_raise_error_for_more_than_one_input_files_paths(self):
        input_collection = Mock()
        input_collection.get_file_paths.return_value = [sentinel.imap_l2_map_path_1, sentinel.imap_l2_map_path_2]

        with self.assertRaises(ValueError) as e:
            UltraL3Dependencies.fetch_dependencies(input_collection)

        self.assertEqual("Incorrect number of dependencies", str(e.exception))

    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.HealPixIntensityMapData.read_from_xarray')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraGlowsL3eData.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.UltraL1CPSet.read_from_path')
    @patch('imap_l3_processing.ultra.l3.ultra_l3_dependencies.ultra_l2')
    def test_from_file_paths(self, mock_ultra_l2, mock_read_l1c: Mock, mock_read_glows: Mock, mock_read_l2: Mock):
        ultra_l2_input_path = Path("um_path")
        ultra_l1c_input_paths = [Path("u_path_1"), Path("u_path_2")]
        glows_input_paths = [Path("g_path_1"), Path("g_path_2")]
        mock_ultra_l2.return_value = [sentinel.ultra_l2_healpix_map]
        mock_read_l1c.side_effect = [sentinel.ultra_data1, sentinel.ultra_data2]
        mock_read_glows.side_effect = [sentinel.glows_data1, sentinel.glows_data2]
        mock_read_l2.return_value = sentinel.ultra_l2_data

        result = UltraL3Dependencies.from_file_paths(ultra_l2_input_path, ultra_l1c_input_paths, glows_input_paths)

        mock_read_l2.assert_called_with(sentinel.ultra_l2_healpix_map)
        mock_read_l1c.assert_has_calls([call(file_path) for file_path in ultra_l1c_input_paths])
        mock_read_glows.assert_has_calls([call(file_path) for file_path in glows_input_paths])

        self.assertEqual(result.ultra_l1c_pset, [sentinel.ultra_data1, sentinel.ultra_data2])
        self.assertEqual(result.glows_l3e_sp, [sentinel.glows_data1, sentinel.glows_data2])
        self.assertEqual(result.ultra_l2_map, sentinel.ultra_l2_data)
