import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call
from zipfile import ZIP_DEFLATED

import numpy as np
from astropy.time import Time, TimeDelta
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess
from imap_l3_processing.glows.l3bc.utils import read_glows_l3a_data, find_unprocessed_carrington_rotations, \
    archive_dependencies, make_l3b_data_with_fill, make_l3c_data_with_fill, get_repoint_date_range
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path, environment_variables

REPOINT_DATA_FILEPATH = get_test_data_path("fake_repointing_file.csv")


class TestUtils(unittest.TestCase):
    def test_read_glows_l3a_data(self):
        cdf = CDF(str(get_test_data_path("glows/imap_glows_l3a_hist_20100101_v001.cdf")))

        actual_glows_lightcurve = read_glows_l3a_data(cdf)

        self.assertAlmostEqual(7.48702879e+01, actual_glows_lightcurve.latitude[0][0])
        self.assertAlmostEqual(154.67118388, actual_glows_lightcurve.longitude[0][0])
        self.assertEqual(datetime(2013, 9, 8, 18, 55, 14), actual_glows_lightcurve.epoch[0])
        self.assertEqual(timedelta(seconds=36180), actual_glows_lightcurve.epoch_delta[0])
        self.assertEqual(802.8, actual_glows_lightcurve.exposure_times[0][0])
        self.assertEqual(0.0007200144163580106, actual_glows_lightcurve.extra_heliospheric_background[0][0])
        self.assertEqual(-27.84000015258789, actual_glows_lightcurve.filter_temperature_average[0])
        self.assertEqual(0.0, actual_glows_lightcurve.filter_temperature_std_dev[0])
        self.assertEqual(1527.0999755859375, actual_glows_lightcurve.hv_voltage_average[0])
        self.assertEqual(87.94999694824219, actual_glows_lightcurve.hv_voltage_std_dev[0])
        self.assertEqual(620.9317389138017, actual_glows_lightcurve.photon_flux[0][0])
        self.assertEqual(0.8794643666117251, actual_glows_lightcurve.photon_flux_uncertainty[0][0])
        self.assertEqual(91.5780029296875, actual_glows_lightcurve.position_angle_offset_average[0])
        self.assertEqual(0.009991000406444073, actual_glows_lightcurve.position_angle_offset_std_dev[0])
        self.assertEqual(0.29899999499320984, actual_glows_lightcurve.pulse_length_average[0])
        self.assertEqual(0.017260000109672546, actual_glows_lightcurve.pulse_length_std_dev[0])
        self.assertEqual(498484, actual_glows_lightcurve.raw_histogram[0][0])
        self.assertEqual(146231104.0, actual_glows_lightcurve.spacecraft_location_average[0][0])
        self.assertEqual(142100.0, actual_glows_lightcurve.spacecraft_location_std_dev[0][0])
        self.assertEqual(6.669000148773193, actual_glows_lightcurve.spacecraft_velocity_average[0][0])
        self.assertEqual(0.11879999935626984, actual_glows_lightcurve.spacecraft_velocity_std_dev[0][0])
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle[0])
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle_delta[0])
        self.assertEqual(162.0919952392578, actual_glows_lightcurve.spin_axis_orientation_average[0][0])
        self.assertEqual(0.2345000058412552, actual_glows_lightcurve.spin_axis_orientation_std_dev[0][0])
        self.assertEqual(15.0, actual_glows_lightcurve.spin_period_average[0])
        self.assertEqual(15.236681938171387, actual_glows_lightcurve.spin_period_ground_average[0])
        self.assertEqual(0.0014979999978095293, actual_glows_lightcurve.spin_period_ground_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.spin_period_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.time_dependent_background[0][0])

    @environment_variables({"REPOINT_DATA_FILEPATH": REPOINT_DATA_FILEPATH})
    def test_get_repoint_date_range(self):
        repointing_number = 12
        actual_start, actual_end = get_repoint_date_range(repointing_number)
        expected_start = np.datetime64(datetime(year=2010, month=1, day=13).isoformat())
        expected_end = np.datetime64(datetime(year=2010, month=1, day=13, hour=23, minute=30).isoformat())

        np.testing.assert_array_equal(actual_start, expected_start)
        np.testing.assert_array_equal(actual_end, expected_end)

    @patch("imap_l3_processing.glows.l3bc.utils.validate_dependencies")
    def test_find_unprocessed_carrington_rotations(self, mock_validate_dependencies: Mock):
        l3a_files_january = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201001{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201001{str(i).zfill(2)}') for i in range(4, 32)
        ]
        l3a_files_february = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201002{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201002{str(i).zfill(2)}') for i in range(1, 29)
        ]
        l3a_files_march = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201003{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201003{str(i).zfill(2)}') for i in range(1, 27)
        ]

        l3a_files_april = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_20100403_v001.pkts',
                data_level='l3a', start_date=f'20100403'),
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_20100423_v001.pkts',
                data_level='l3a', start_date=f'20100423'),
        ]
        l3a_files_june = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_2010711_v001.pkts',
                data_level='l3a', start_date=f'20100711'),
        ]

        l3a_files = l3a_files_february + l3a_files_march + l3a_files_january + l3a_files_april + l3a_files_june

        l3b_files = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3bc/2010/01/imap_glows_l3b_hist_20100130_v001.pkts',
                data_level='l3b', start_date=f'20100130')
        ]

        mock_validate_dependencies.side_effect = [True, False, True, True]

        expected_l3a_2092 = [create_l3a_path_by_date(f'201001{str(i).zfill(2)}') for i in range(4, 31)]

        expected_l3a_2095 = [create_l3a_path_by_date('20100326'), create_l3a_path_by_date('20100403')]

        expected_l3a_2096 = [create_l3a_path_by_date('20100423')]

        initializer_dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                                         waw_helioion_mp_path="waw_helioion",
                                                                         bad_days_list="bad_days_list",
                                                                         pipeline_settings="pipeline_settings",
                                                                         lyman_alpha_path=Path("lyman_alpha"),
                                                                         omni2_data_path=Path("omni"),
                                                                         initializer_time_buffer=TimeDelta(52,
                                                                                                           format="jd"),
                                                                         f107_index_file_path=Path("f107"))

        actual_crs_to_process: [CRToProcess] = find_unprocessed_carrington_rotations(l3a_files, l3b_files,
                                                                                     initializer_dependencies)

        self.assertEqual(3, len(actual_crs_to_process))
        cr_to_process_2092 = actual_crs_to_process[0]
        self.assertEqual(expected_l3a_2092, cr_to_process_2092.l3a_paths)
        self.assertEqual(Time('2010-01-03 11:33:04.320').value, cr_to_process_2092.cr_start_date.value)
        self.assertEqual(Time('2010-01-30 18:09:30.240').value, cr_to_process_2092.cr_end_date.value)
        self.assertEqual(2092, cr_to_process_2092.cr_rotation_number)

        cr_to_process_2095 = actual_crs_to_process[1]
        self.assertEqual(expected_l3a_2095, cr_to_process_2095.l3a_paths)
        self.assertEqual(Time('2010-03-26 07:22:22.080').value, cr_to_process_2095.cr_start_date.value)
        self.assertEqual(Time('2010-04-22 13:58:48.000').value, cr_to_process_2095.cr_end_date.value)
        self.assertEqual(2095, cr_to_process_2095.cr_rotation_number)

        cr_to_process_2096 = actual_crs_to_process[2]
        self.assertEqual(expected_l3a_2096, cr_to_process_2096.l3a_paths)
        self.assertEqual(Time('2010-04-22 13:58:48.000').value, cr_to_process_2096.cr_start_date.value)
        self.assertEqual(Time('2010-05-19 20:35:13.920').value, cr_to_process_2096.cr_end_date.value)
        self.assertEqual(2096, cr_to_process_2096.cr_rotation_number)

        self.assertEqual(Time('2010-01-30 18:09:30.240').value,
                         mock_validate_dependencies.call_args_list[0][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[0][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[0][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[0][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[0][0][4])

        self.assertEqual(Time('2010-03-26 07:22:22.080').value,
                         mock_validate_dependencies.call_args_list[1][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[1][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[1][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[1][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[1][0][4])

        self.assertEqual(Time('2010-04-22 13:58:48.000').value,
                         mock_validate_dependencies.call_args_list[2][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[2][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[2][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[2][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[2][0][4])

        self.assertEqual(Time('2010-05-19 20:35:13.920').value,
                         mock_validate_dependencies.call_args_list[3][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[3][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[3][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[3][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[3][0][4])

    @patch("imap_l3_processing.glows.l3bc.utils.json")
    @patch("imap_l3_processing.glows.l3bc.utils.ZipFile")
    def test_archive_dependencies(self, mock_zip, mock_json):
        expected_filepath = TEMP_CDF_FOLDER_PATH / "imap_glows_l3b-archive-zip_20250314_v001.cdf"
        expected_json_filename = "cr_to_process.json"

        dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                             waw_helioion_mp_path="waw_helioion",
                                                             bad_days_list="bad_days",
                                                             pipeline_settings="pipeline_settings",
                                                             lyman_alpha_path=Path("lyman_alpha"),
                                                             omni2_data_path=Path("omni"),
                                                             f107_index_file_path=Path("f107"),
                                                             initializer_time_buffer=TimeDelta(42, format="jd"),
                                                             )

        cr_to_process: CRToProcess = CRToProcess(cr_rotation_number=2095, l3a_paths=["file1", "file2"],
                                                 cr_start_date=Time("2025-03-14 12:34:56.789"),
                                                 cr_end_date=Time("2025-03-24 12:34:56.789"))

        expected_json_to_serialize = {"cr_rotation_number": 2095,
                                      "l3a_paths": ["file1", "file2"],
                                      "cr_start_date": "2025-03-14 12:34:56.789",
                                      "cr_end_date": "2025-03-24 12:34:56.789",
                                      "bad_days_list": dependencies.bad_days_list,
                                      "pipeline_settings": dependencies.pipeline_settings,
                                      "waw_helioion_mp": dependencies.waw_helioion_mp_path,
                                      "uv_anisotropy": dependencies.uv_anisotropy_path
                                      }

        mock_zip_file = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_file

        version_number = "v001"
        actual_zip_file_name = archive_dependencies(cr_to_process, version_number, dependencies)

        self.assertEqual(expected_filepath, actual_zip_file_name)

        mock_zip.assert_called_with(expected_filepath, "w", ZIP_DEFLATED)

        mock_json.dumps.assert_called_once_with(expected_json_to_serialize)

        mock_zip_file.write.assert_has_calls([
            call(dependencies.lyman_alpha_path, "lyman_alpha_composite.nc"),
            call(dependencies.omni2_data_path, "omni2_all_years.dat"),
            call(dependencies.f107_index_file_path, "f107_fluxtable.txt"),
        ])
        mock_zip_file.writestr.assert_called_once_with(expected_json_filename, mock_json.dumps.return_value)

    def test_make_l3b_data_with_fill(self):
        cr = 2091
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'pipeline_settings': get_test_instrument_team_data_path(
                'glows/imap_glows_pipeline-settings-L3bc_v001.json'),
        }
        external_files = Mock()

        dependencies = GlowsL3BCDependencies(l3a_data=Mock(), external_files=external_files,
                                             ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                             start_date=datetime(2025, 1, 1),
                                             end_date=datetime(2025, 1, 3),
                                             zip_file_path=Path("file.zip"))
        l3b_data_with_fill = make_l3b_data_with_fill(dependencies)

        self.assertEqual({
            'ancillary_data_files': ancillary_files,
            'external_dependeciens': external_files,
            'l3a_input_files_name': [],
        }, l3b_data_with_fill['header'])

        num_lat_grid_values = 19
        self.assertEqual([-90, -80, -70, -60, -50, -40, -30, -20, -10,
                          0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                         l3b_data_with_fill['ion_rate_profile']['lat_grid'])

        self.assertEqual(2091, l3b_data_with_fill['CR'])

        np.testing.assert_array_equal(l3b_data_with_fill['uv_anisotropy_factor'],
                                      np.full(num_lat_grid_values, 1))

        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['sum_rate'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['ph_rate'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['cx_rate'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['sum_uncert'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['ph_uncert'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3b_data_with_fill['ion_rate_profile']['cx_uncert'],
                                      np.full(num_lat_grid_values, np.nan))

    def test_make_l3c_data_with_fill(self):
        cr = 2091
        ancillary_files = {
            'uv_anisotropy': get_test_data_path('glows/imap_glows_uv-anisotropy-1CR_20100101_v001.json'),
            'pipeline_settings': get_test_instrument_team_data_path(
                'glows/imap_glows_pipeline-settings-L3bc_v001.json'),
        }
        external_files = Mock()

        dependencies = GlowsL3BCDependencies(l3a_data=Mock(), external_files=external_files,
                                             ancillary_files=ancillary_files, carrington_rotation_number=cr,
                                             start_date=datetime(2025, 1, 1),
                                             end_date=datetime(2025, 1, 3),
                                             zip_file_path=Path("file.zip"))
        l3c_data_with_fill = make_l3c_data_with_fill(dependencies)

        self.assertEqual({
            'ancillary_data_files': ancillary_files,
            'external_dependeciens': external_files,
        }, l3c_data_with_fill['header'])

        num_lat_grid_values = 19

        self.assertEqual(2091, l3c_data_with_fill['CR'])

        np.testing.assert_array_equal(l3c_data_with_fill['solar_wind_ecliptic']['plasma_speed'], np.nan)
        np.testing.assert_array_equal(l3c_data_with_fill['solar_wind_ecliptic']['proton_density'], np.nan)
        np.testing.assert_array_equal(l3c_data_with_fill['solar_wind_ecliptic']['alpha_abundance'], np.nan)
        self.assertEqual([-90, -80, -70, -60, -50, -40, -30, -20, -10,
                          0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                         l3c_data_with_fill['solar_wind_profile']['lat_grid'])
        np.testing.assert_array_equal(l3c_data_with_fill['solar_wind_profile']['plasma_speed'],
                                      np.full(num_lat_grid_values, np.nan))
        np.testing.assert_array_equal(l3c_data_with_fill['solar_wind_profile']['proton_density'],
                                      np.full(num_lat_grid_values, np.nan))


def create_imap_data_access_json(file_path: str, data_level: str, start_date: str,
                                 descriptor: str = "hist", version: str = "v001") -> dict:
    return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': descriptor,
            'start_date': start_date, 'repointing': None, 'version': version, 'extension': 'pkts',
            'ingestion_date': '2024-10-11 15:28:32'}


def create_l3a_path_by_date(file_date: str) -> str:
    return f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_{file_date}_v001.pkts'
