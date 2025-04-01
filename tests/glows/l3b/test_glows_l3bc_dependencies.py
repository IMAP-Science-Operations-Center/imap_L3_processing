import unittest
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock, MagicMock, mock_open

from imap_l3_processing.glows.l3b import glows_l3bc_dependencies
from imap_l3_processing.glows.l3b.glows_l3bc_dependencies import GlowsL3BCDependencies


class TestGlowsL3BCDependencies(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.l3b.glows_l3bc_dependencies.ZipFile')
    @patch('imap_l3_processing.glows.l3b.glows_l3bc_dependencies.CDF')
    @patch('imap_l3_processing.glows.l3b.glows_l3bc_dependencies.download_dependency')
    @patch('imap_l3_processing.glows.l3b.glows_l3bc_dependencies.download_dependency_from_path')
    def test_fetch_dependencies(self, mock_download_dependencies_from_path, mock_download_dependencies,
                                mock_cdf_constructor, mock_zip_file_class,
                                mock_open_file):
        mock_zip_file_path = Mock()
        mock_download_dependencies.side_effect = [mock_zip_file_path]
        mock_download_dependencies_from_path.side_effect = [
            sentinel.uv_anisotropy_downloaded_path,
            sentinel.waw_downloaded_path
        ]

        mock_zip_file = MagicMock()
        mock_zip_file_class.return_value.__enter__.return_value = mock_zip_file

        mock_json_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = mock_json_file
        mock_json_file.read.return_value = '{"l3a_paths":["l3a_path_1", "l3a_path_2"], "uv_anisotropy":"uv_anisotropy_path", "waw_helioion_mp":"waw_path"}'

        dependency: GlowsL3BCDependencies = GlowsL3BCDependencies.fetch_dependencies([sentinel.zip_dependency])

        mock_zip_file_class.assert_called_with(mock_zip_file_path, 'r')
        base_dir_file_path = Path(glows_l3bc_dependencies.__file__).parent / 'glows_l3b_files'
        mock_zip_file.extract.assert_called_once_with(str(base_dir_file_path))

        mock_download_dependencies.assert_called_once_with(sentinel.zip_dependency)

        self.assertEqual([call("uv_anisotropy_path"),
                          call("waw_path")
                          ], mock_download_dependencies_from_path.call_args_list)

        self.assertEqual(base_dir_file_path / 'f107_fluxtable.txt', dependency.external_files['f107_raw_data'])
        self.assertEqual(base_dir_file_path / 'omni2_all_years.dat', dependency.external_files['omni_raw_data'])

        self.assertEqual(sentinel.uv_anisotropy_downloaded_path, dependency.ancillary_files['uv_anisotropy'])
        self.assertEqual(sentinel.waw_downloaded_path, dependency.ancillary_files['waw_helioion_mp'])
