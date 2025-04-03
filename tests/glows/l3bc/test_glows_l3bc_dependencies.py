import unittest
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock, MagicMock, mock_open

from imap_l3_processing.glows.l3bc import glows_l3bc_dependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies


class TestGlowsL3BCDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.create_glows_l3a_dictionary_from_cdf')
    @patch('builtins.open', new_callable=mock_open, create=False)
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.ZipFile')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.download_dependency_from_path')
    def test_fetch_dependencies(self, mock_download_dependencies_from_path,
                                mock_zip_file_class,
                                mock_open_file, mock_create_dictionary_from_cdf):
        mock_zip_file_path = Mock()
        mock_download_dependencies_from_path.side_effect = [
            sentinel.uv_anisotropy_downloaded_path,
            sentinel.waw_downloaded_path,
            sentinel.bad_day_list_downloaded_path,
            sentinel.settings_downloaded_path,
            sentinel.l3a_downloaded_path_1,
            sentinel.l3a_downloaded_path_2,
        ]

        mock_zip_file = MagicMock()
        mock_zip_file_class.return_value.__enter__.return_value = mock_zip_file

        mock_json_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = mock_json_file
        mock_json_file.read.return_value = '{"l3a_paths":["l3a_path_1", "l3a_path_2"],' \
                                           '"uv_anisotropy":"uv_anisotropy_path", "waw_helioion_mp":"waw_path",' \
                                           '"bad_days_list":"bad_days_list_path", "pipeline_settings":"pipeline_settings_path",' \
                                           '"cr_rotation_number":"2296"}'

        mock_create_dictionary_from_cdf.side_effect = [
            sentinel.l3a_dictionary_1,
            sentinel.l3a_dictionary_2,
        ]

        dependency: GlowsL3BCDependencies = GlowsL3BCDependencies.fetch_dependencies(mock_zip_file_path)

        mock_zip_file_class.assert_called_with(mock_zip_file_path, 'r')
        base_dir_file_path = Path(glows_l3bc_dependencies.__file__).parent / 'glows_l3b_files'
        mock_zip_file.extract.assert_called_once_with(str(base_dir_file_path))

        self.assertEqual([call(sentinel.l3a_downloaded_path_1), call(sentinel.l3a_downloaded_path_2)],
                         mock_create_dictionary_from_cdf.call_args_list)
        self.assertEqual([call("uv_anisotropy_path"),
                          call("waw_path"),
                          call("bad_days_list_path"),
                          call("pipeline_settings_path"),
                          call("l3a_path_1"),
                          call("l3a_path_2"),
                          ], mock_download_dependencies_from_path.call_args_list)

        self.assertEqual(base_dir_file_path / 'f107_fluxtable.txt', dependency.external_files['f107_raw_data'])
        self.assertEqual(base_dir_file_path / 'omni2_all_years.dat', dependency.external_files['omni_raw_data'])

        self.assertEqual(sentinel.uv_anisotropy_downloaded_path, dependency.ancillary_files['uv_anisotropy'])
        self.assertEqual(sentinel.waw_downloaded_path, dependency.ancillary_files['WawHelioIonMP_parameters'])
        self.assertEqual(sentinel.bad_day_list_downloaded_path, dependency.ancillary_files['bad_days_list'])
        self.assertEqual(sentinel.settings_downloaded_path, dependency.ancillary_files['pipeline_settings'])

        self.assertEqual(sentinel.l3a_dictionary_1, dependency.l3a_data[0])
        self.assertEqual(sentinel.l3a_dictionary_2, dependency.l3a_data[1])
        self.assertEqual(2296, dependency.carrington_rotation_number)
