import unittest
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import patch, Mock, MagicMock, call
from zipfile import ZIP_DEFLATED

import numpy as np
from astropy.time import Time, TimeDelta
from imap_processing.spice.repoint import set_global_repoint_table_paths
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess
from imap_l3_processing.glows.l3bc.utils import read_glows_l3a_data, find_unprocessed_carrington_rotations, \
    archive_dependencies, get_pointing_date_range
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repoint_file_path = get_test_data_path("fake_1_day_repointing_file.csv")
        set_global_repoint_table_paths([repoint_file_path])

    def test_read_glows_l3a_data(self):
        cdf = CDF(str(get_test_data_path("glows/imap_glows_l3a_hist_20100101_v001.cdf")))

        actual_glows_lightcurve = read_glows_l3a_data(cdf)

        self.assertAlmostEqual(7.48702879e+01, actual_glows_lightcurve.latitude[0][0])
        self.assertAlmostEqual(154.67118388, actual_glows_lightcurve.longitude[0][0])
        self.assertEqual(datetime(2013, 9, 8, 18, 55, 14), actual_glows_lightcurve.epoch[0])
        self.assertEqual(36180, actual_glows_lightcurve.epoch_delta[0])
        self.assertEqual(802.8, actual_glows_lightcurve.exposure_times[0][0])
        self.assertEqual(65, actual_glows_lightcurve.number_of_bins)
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

    def test_get_pointing_date_range(self):
        repointing_number = 13
        actual_start, actual_end = get_pointing_date_range(repointing_number)
        expected_start = np.datetime64(
            datetime(year=2000, month=1, day=14, hour=12, minute=13, second=55, microsecond=816000).isoformat())
        expected_end = np.datetime64(
            datetime(year=2000, month=1, day=15, hour=11, minute=58, second=55, microsecond=816000).isoformat())

        np.testing.assert_array_equal(actual_start, expected_start)
        np.testing.assert_array_equal(actual_end, expected_end)

    def test_get_repoint_date_range_handles_no_pointing(self):
        repointing_number = 5998
        with self.assertRaises(ValueError) as err:
            _, _ = get_pointing_date_range(repointing_number)
        self.assertEqual(str(err.exception), f"No pointing found for pointing: 5998")

    @patch("imap_l3_processing.glows.l3bc.utils.validate_dependencies")
    def test_find_unprocessed_carrington_rotations(self,
                                                   mock_validate_dependencies: Mock):
        set_global_repoint_table_paths([Path("not_set_yet")])
        l3a_files_january = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200001{str(i + 1).zfill(2)}-repoint{str(i).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200001{str(i + 1).zfill(2)}', repointing=i) for i in range(4, 31)
        ]
        l3a_files_february = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200002{str(i + 1).zfill(2)}-repoint{str(i + 31).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200002{str(i + 1).zfill(2)}', repointing=i + 31) for i in range(1, 29)
        ]
        l3a_files_march = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_200003{str(i + 1).zfill(2)}-repoint{str(i + 60).zfill(5)}_v001.pkts',
                data_level='l3a', start_date=f'200003{str(i + 1).zfill(2)}', repointing=i + 60) for i in range(1, 27)
        ]

        l3a_files_april = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000403-repoint00093_v001.pkts',
                data_level='l3a', start_date=f'20000403', repointing=93),
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000423-repoint00113_v001.pkts',
                data_level='l3a', start_date=f'20000423', repointing=113),
        ]
        l3a_files_july = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000720-repoint00201_v001.pkts',
                data_level='l3a', start_date=f'20000720', repointing=201),
        ]

        l3a_files = l3a_files_february + l3a_files_march + l3a_files_january + l3a_files_april + l3a_files_july

        l3b_files = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3bc/2000/01/imap_glows_l3b_hist_20000130_v001.pkts',
                data_level='l3b', start_date=f'20000130')
        ]

        mock_validate_dependencies.side_effect = [True, False, True, True]

        expected_l3a_1958 = [create_l3a_path_by_date(f'200001{str(i + 1).zfill(2)}', i) for i in range(4, 28)]
        expected_l3a_1959 = [create_l3a_path_by_date(f'200001{str(i + 1).zfill(2)}', i) for i in range(27, 31)]
        expected_l3a_1959 += [create_l3a_path_by_date(f'200002{str(i + 1).zfill(2)}', i + 31) for i in range(1, 24)]
        # expected_l3a_1960 = [create_l3a_path_by_date(f'200002{str(i + 1).zfill(2)}', i + 31) for i in range(24, 29)]
        # expected_l3a_1960 += [create_l3a_path_by_date(f'200003{str(i).zfill(2)}', i + 60) for i in range(1, 24)]
        expected_l3a_1961 = [create_l3a_path_by_date(f'200003{str(i + 1).zfill(2)}', i + 60) for i in range(21, 27)]
        expected_l3a_1961.append(create_l3a_path_by_date(f'20000403', 93))
        expected_l3a_1962 = [create_l3a_path_by_date(f'20000423', 113)]
        # expected_l3a_1965 = [create_l3a_path_by_date(f'20000711', 192)]

        initializer_dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                                         waw_helioion_mp_path="waw_helioion",
                                                                         bad_days_list="bad_days_list",
                                                                         pipeline_settings="pipeline_settings",
                                                                         lyman_alpha_path=Path("lyman_alpha"),
                                                                         omni2_data_path=Path("omni"),
                                                                         initializer_time_buffer=TimeDelta(52,
                                                                                                           format="jd"),
                                                                         f107_index_file_path=Path("f107"),
                                                                         repointing_file=get_test_data_path(
                                                                             "fake_1_day_repointing_file.csv"))

        actual_crs_to_process: [CRToProcess] = find_unprocessed_carrington_rotations(l3a_files, l3b_files,
                                                                                     initializer_dependencies)

        self.assertEqual(3, len(actual_crs_to_process))
        cr_to_process_1958 = actual_crs_to_process[0]
        self.assertEqual(expected_l3a_1958, cr_to_process_1958.l3a_paths)
        self.assertEqual(Time('2000-01-01 14:11:11.040').value, cr_to_process_1958.cr_start_date.value)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1958.cr_end_date.value)
        self.assertEqual(1958, cr_to_process_1958.cr_rotation_number)

        cr_to_process_1961 = actual_crs_to_process[1]
        self.assertEqual(expected_l3a_1961, cr_to_process_1961.l3a_paths)
        self.assertEqual(Time('2000-03-23 10:00:28.800').value, cr_to_process_1961.cr_start_date.value)
        self.assertEqual(Time('2000-04-19 16:36:54.720').value, cr_to_process_1961.cr_end_date.value)
        self.assertEqual(1961, cr_to_process_1961.cr_rotation_number)

        cr_to_process_1962 = actual_crs_to_process[2]
        self.assertEqual(expected_l3a_1962, cr_to_process_1962.l3a_paths)
        self.assertEqual(Time('2000-04-19 16:36:54.720').value, cr_to_process_1962.cr_start_date.value)
        self.assertEqual(Time('2000-05-16 23:13:20.640').value, cr_to_process_1962.cr_end_date.value)
        self.assertEqual(1962, cr_to_process_1962.cr_rotation_number)

        self.assertEqual(Time('2000-01-28 20:47:36.960').value,
                         mock_validate_dependencies.call_args_list[0][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[0][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[0][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[0][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[0][0][4])

        self.assertEqual(Time('2000-03-23 10:00:28.800').value,
                         mock_validate_dependencies.call_args_list[1][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[1][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[1][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[1][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[1][0][4])

        self.assertEqual(Time('2000-04-19 16:36:54.720').value,
                         mock_validate_dependencies.call_args_list[2][0][0].value)
        self.assertEqual(initializer_dependencies.initializer_time_buffer.value,
                         mock_validate_dependencies.call_args_list[2][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[2][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[2][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[2][0][4])

    @patch("imap_l3_processing.glows.l3bc.utils.validate_dependencies")
    def test_find_unprocessed_carrington_rotations_handles_multi_day_repointing(self, mock_validate_dependencies: Mock):
        l3a_in_1958 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000104-repoint00004_v001.cdf',
            data_level='l3a',
            start_date=f'20000104', repointing=4)
        l3a_in_1958_and_1959 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20000128-repoint00027_v001.cdf',
            data_level='l3a',
            start_date=f'20000128', repointing=27)
        l3a_in_1959 = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_20001031-repoint00030_v001.cdf',
            data_level='l3a',
            start_date=f'20000131', repointing=30)
        l3a_past_buffer_range = create_imap_data_access_json(
            file_path='imap/glows/l3a/2000/01/imap_glows_l3a_hist_2000528-repoint00148_v001.cdf',
            data_level='l3a',
            start_date=f'20000528', repointing=148)
        l3a_files = [l3a_in_1958, l3a_in_1958_and_1959, l3a_in_1959, l3a_past_buffer_range]

        mock_validate_dependencies.return_value = True

        expected_l3a_1958 = [l3a_in_1958.get('file_path'), l3a_in_1958_and_1959.get('file_path')]
        expected_l3a_1959 = [l3a_in_1958_and_1959.get('file_path'), l3a_in_1959.get('file_path')]

        mock_dependencies = Mock(initializer_time_buffer=56,
                                 repointing_file=get_test_data_path("fake_1_day_repointing_file.csv"))
        actual_crs_to_process: [CRToProcess] = find_unprocessed_carrington_rotations(l3a_files, [], mock_dependencies)

        self.assertEqual(2, len(actual_crs_to_process))
        cr_to_process_1958 = actual_crs_to_process[0]
        self.assertEqual(expected_l3a_1958, cr_to_process_1958.l3a_paths)
        self.assertEqual(Time('2000-01-01 14:11:11.040').value, cr_to_process_1958.cr_start_date.value)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1958.cr_end_date.value)
        self.assertEqual(1958, cr_to_process_1958.cr_rotation_number)

        cr_to_process_1959 = actual_crs_to_process[1]
        self.assertEqual(expected_l3a_1959, cr_to_process_1959.l3a_paths)
        self.assertEqual(Time('2000-01-28 20:47:36.960').value, cr_to_process_1959.cr_start_date.value)
        self.assertEqual(Time('2000-02-25 03:24:02.880').value, cr_to_process_1959.cr_end_date.value)
        self.assertEqual(1959, cr_to_process_1959.cr_rotation_number)

    @patch("imap_l3_processing.glows.l3bc.utils.json")
    @patch("imap_l3_processing.glows.l3bc.utils.ZipFile")
    def test_archive_dependencies(self, mock_zip, mock_json):
        expected_filepath = TEMP_CDF_FOLDER_PATH / "imap_glows_l3b-archive_20250314_v001.zip"
        expected_json_filename = "cr_to_process.json"

        dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                             waw_helioion_mp_path="waw_helioion",
                                                             bad_days_list="bad_days",
                                                             pipeline_settings="pipeline_settings",
                                                             lyman_alpha_path=Path("lyman_alpha"),
                                                             omni2_data_path=Path("omni"),
                                                             f107_index_file_path=Path("f107"),
                                                             initializer_time_buffer=TimeDelta(42, format="jd"),
                                                             repointing_file=Path("/path/to/repointing.csv")
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
                                      "uv_anisotropy": dependencies.uv_anisotropy_path,
                                      "repointing_file": "repointing.csv",
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


def create_imap_data_access_json(file_path: str, data_level: str, start_date: str,
                                 descriptor: str = "hist", version: str = "v001",
                                 repointing: Optional[int] = None) -> dict:
    return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': descriptor,
            'start_date': start_date, 'repointing': repointing, 'version': version, 'extension': 'pkts',
            'ingestion_date': '2024-10-11 15:28:32'}


def create_l3a_path_by_date(file_date: str, repointing: int) -> str:
    return f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_{file_date}-repoint{str(repointing).zfill(5)}_v001.pkts'
