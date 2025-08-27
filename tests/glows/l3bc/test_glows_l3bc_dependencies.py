import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, sentinel, call

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess, ExternalDependencies


class TestGlowsL3BCDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.create_glows_l3a_dictionary_from_cdf')
    @patch('imap_l3_processing.glows.l3bc.glows_l3bc_dependencies.imap_data_access.download')
    def test_download_from_cr_to_process(self, mock_download, mock_create_dictionary_from_cdf):
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
            cr_rotation_number=2296,
        )

        external_dependencies = ExternalDependencies(
            f107_index_file_path=TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt',
            omni2_data_path=TEMP_CDF_FOLDER_PATH / 'omni2_all_years.dat',
            lyman_alpha_path=Path("not used"),
        )

        dependency: GlowsL3BCDependencies = GlowsL3BCDependencies.download_from_cr_to_process(cr_to_process,
                                                                                              sentinel.version,
                                                                                              external_dependencies)

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

        self.assertEqual(sentinel.version, dependency.version)

        self.assertEqual(cr_to_process.cr_start_date, dependency.start_date)
        self.assertEqual(cr_to_process.cr_end_date, dependency.end_date)
