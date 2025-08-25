import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, sentinel, call, Mock, MagicMock, mock_open

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess
from tests.test_helpers import get_test_data_path


class TestGlowsL3BCDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.create_glows_l3a_dictionary_from_cdf')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.imap_data_access.download')
    def test_from_cr_to_process(self, mock_download, mock_create_dictionary_from_cdf):
        mock_download.side_effect = [
            sentinel.uv_anisotropy_downloaded_path,
            sentinel.waw_downloaded_path,
            sentinel.bad_day_list_downloaded_path,
            sentinel.settings_downloaded_path,
            sentinel.l3a_downloaded_path_1,
            sentinel.l3a_downloaded_path_2,
        ]

        mock_create_dictionary_from_cdf.side_effect = [
            sentinel.l3a_dictionary_1,
            sentinel.l3a_dictionary_2,
        ]

        cr_to_process = CRToProcess(
            l3a_file_names={"l3a_path_1", "l3a_path_2"},
            uv_anisotropy_file_name="uv_anisotropy_path",
            waw_helio_ion_mp_file_name="waw_path",
            bad_days_list_file_name="bad_days_list_path",
            pipeline_settings_file_name="pipeline_settings_path",
            cr_start_date=datetime.fromisoformat("2024-10-24 12:34:56.789"),
            cr_end_date=datetime.fromisoformat("2024-11-24 12:34:56.789"),
            f107_index_file_path=TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt',
            omni2_data_path=TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat',
            lyman_alpha_path=Path("not used"),
            cr_rotation_number=2296,
        )

        dependency: GlowsL3BCDependencies = GlowsL3BCDependencies.from_cr_to_process(cr_to_process)

        self.assertEqual([call(sentinel.l3a_downloaded_path_1), call(sentinel.l3a_downloaded_path_2)],
                         mock_create_dictionary_from_cdf.call_args_list)

        mock_download.assert_has_calls([
            call("uv_anisotropy_path"),
            call("waw_path"),
            call("bad_days_list_path"),
            call("pipeline_settings_path"),
            call("l3a_path_1"),
            call("l3a_path_2")
        ])

        self.assertEqual(TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt', dependency.external_files['f107_raw_data'])
        self.assertEqual(TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat', dependency.external_files['omni_raw_data'])

        self.assertEqual(sentinel.uv_anisotropy_downloaded_path, dependency.ancillary_files['uv_anisotropy'])
        self.assertEqual(sentinel.waw_downloaded_path, dependency.ancillary_files['WawHelioIonMP_parameters'])
        self.assertEqual(sentinel.bad_day_list_downloaded_path, dependency.ancillary_files['bad_days_list'])
        self.assertEqual(sentinel.settings_downloaded_path, dependency.ancillary_files['pipeline_settings'])

        self.assertEqual(sentinel.l3a_dictionary_1, dependency.l3a_data[0])
        self.assertEqual(sentinel.l3a_dictionary_2, dependency.l3a_data[1])
        self.assertEqual(2296, dependency.carrington_rotation_number)

        self.assertEqual(cr_to_process.cr_start_date, dependency.start_date)
        self.assertEqual(cr_to_process.cr_end_date, dependency.end_date)

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
                                           '"cr_rotation_number":"2296", "cr_start_date":"2024-10-24 12:34:56.789",' \
                                           '"cr_end_date":"2024-11-24 12:34:56.789"}'

        mock_create_dictionary_from_cdf.side_effect = [
            sentinel.l3a_dictionary_1,
            sentinel.l3a_dictionary_2,
        ]

        dependency: GlowsL3BCDependencies = GlowsL3BCDependencies.fetch_dependencies(mock_zip_file_path)

        mock_zip_file_class.assert_called_with(mock_zip_file_path, 'r')
        mock_zip_file.extract.assert_has_calls([
            call('f107_fluxtable.txt', TEMP_CDF_FOLDER_PATH),
            call('omni2_all_years.dat', TEMP_CDF_FOLDER_PATH),
            call('cr_to_process.json', TEMP_CDF_FOLDER_PATH),
        ])

        self.assertEqual([call(sentinel.l3a_downloaded_path_1), call(sentinel.l3a_downloaded_path_2)],
                         mock_create_dictionary_from_cdf.call_args_list)
        self.assertEqual([call("uv_anisotropy_path"),
                          call("waw_path"),
                          call("bad_days_list_path"),
                          call("pipeline_settings_path"),
                          call("l3a_path_1"),
                          call("l3a_path_2"),
                          ], mock_download_dependencies_from_path.call_args_list)

        self.assertEqual(TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt', dependency.external_files['f107_raw_data'])
        self.assertEqual(TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat', dependency.external_files['omni_raw_data'])

        self.assertEqual(sentinel.uv_anisotropy_downloaded_path, dependency.ancillary_files['uv_anisotropy'])
        self.assertEqual(sentinel.waw_downloaded_path, dependency.ancillary_files['WawHelioIonMP_parameters'])
        self.assertEqual(sentinel.bad_day_list_downloaded_path, dependency.ancillary_files['bad_days_list'])
        self.assertEqual(sentinel.settings_downloaded_path, dependency.ancillary_files['pipeline_settings'])

        self.assertEqual(sentinel.l3a_dictionary_1, dependency.l3a_data[0])
        self.assertEqual(sentinel.l3a_dictionary_2, dependency.l3a_data[1])
        self.assertEqual(2296, dependency.carrington_rotation_number)

        self.assertEqual(datetime.fromisoformat("2024-10-24 12:34:56.789"), dependency.start_date)
        self.assertEqual(datetime.fromisoformat("2024-11-24 12:34:56.789"), dependency.end_date)

        self.assertEqual(mock_zip_file_path, dependency.zip_file_path)

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.create_glows_l3a_dictionary_from_cdf')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.download_dependency_from_path')
    def test_fetch_dependencies_from_real_zip_file(self, mock_download_dependencies, mock_create_dict):
        zip_file_path = get_test_data_path('glows/imap_glows_l3b-archive_20100103_v002.zip')
        mock_download_dependencies.side_effect = [
            sentinel.uv_anisotropy_downloaded_path,
            sentinel.waw_downloaded_path,
            sentinel.bad_day_list_downloaded_path,
            sentinel.pipeline_settings_downloaded_path,
            sentinel.l3a_downloaded_path_1,
            sentinel.l3a_downloaded_path_2,
        ]
        mock_create_dict.side_effect = [
            sentinel.l3a_dictionary_1,
            sentinel.l3a_dictionary_2
        ]

        dependencies = GlowsL3BCDependencies.fetch_dependencies(zip_file_path)

        expected_external_files = {
            'f107_raw_data': TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt',
            'omni_raw_data': TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat',
        }
        expected_ancillary_files = {
            'uv_anisotropy': sentinel.uv_anisotropy_downloaded_path,
            'WawHelioIonMP_parameters': sentinel.waw_downloaded_path,
            'bad_days_list': sentinel.bad_day_list_downloaded_path,
            'pipeline_settings': sentinel.pipeline_settings_downloaded_path,
        }

        self.assertEqual([sentinel.l3a_dictionary_1, sentinel.l3a_dictionary_2], dependencies.l3a_data)
        self.assertEqual(expected_external_files, dependencies.external_files)
        self.assertEqual(expected_ancillary_files, dependencies.ancillary_files)
        self.assertEqual(2092, dependencies.carrington_rotation_number)
        self.assertEqual(datetime(2010, 1, 3, 11, 33, 4, 320000), dependencies.start_date)
        self.assertEqual(datetime(2010, 1, 30, 18, 9, 30, 240000), dependencies.end_date)
        self.assertEqual(zip_file_path, dependencies.zip_file_path)
