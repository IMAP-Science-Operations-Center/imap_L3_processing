import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call, mock_open
from zipfile import ZIP_DEFLATED

from astropy.time import Time
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3b.models import CRToProcess
from imap_l3_processing.glows.l3b.utils import read_glows_l3a_data, find_unprocessed_carrington_rotations, \
    archive_dependencies
from tests.test_helpers import get_test_data_path


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

    @patch("imap_l3_processing.glows.l3b.utils.validate_dependencies")
    def test_find_unprocessed_carrington_rotations(self, mock_validate_dependencies: Mock):
        l3a_files_january = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201001{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201001{str(i).zfill(2)}') for i in range(1, 32)
        ]
        l3a_files_february = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201002{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201002{str(i).zfill(2)}') for i in range(1, 29)
        ]
        l3a_files_march = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201003{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201003{str(i).zfill(2)}') for i in range(1, 32)
        ]

        l3a_files_april = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201004{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201004{str(i).zfill(2)}') for i in range(1, 31)
        ]

        l3a_files = l3a_files_february + l3a_files_march + l3a_files_january + l3a_files_april

        l3b_files = [
            create_imap_data_access_json(
                file_path=f'imap/glows/l3b/2010/01/imap_glows_l3b_hist_20100213_v001.pkts',
                data_level='l3b', start_date=f'20100213')
        ]

        mock_validate_dependencies.side_effect = [True, False, True]

        expected_l3a_january_paths = [create_l3a_path_by_date(f'201001{str(i).zfill(2)}') for i in range(3, 31)]

        expected_l3a_april_paths = [create_l3a_path_by_date(f'201003{str(i).zfill(2)}') for i in range(26, 32)] + [
            create_l3a_path_by_date(f'201004{str(i).zfill(2)}') for i in range(1, 23)]

        initializer_dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                                         lyman_alpha_path=Path("lyman_alpha"),
                                                                         omni2_data_path=Path("omni"),
                                                                         f107_index_file_path=Path("f107"),
                                                                         waw_helioion_mp_path="waw_helioion")

        actual_crs_to_process: [CRToProcess] = find_unprocessed_carrington_rotations(l3a_files, l3b_files,
                                                                                     initializer_dependencies)

        self.assertEqual(2, len(actual_crs_to_process))
        self.assertEqual(expected_l3a_january_paths, actual_crs_to_process[0].l3a_paths)
        self.assertEqual('20100117', actual_crs_to_process[0].cr_midpoint)
        self.assertEqual(2092, actual_crs_to_process[0].cr_rotation_number)
        self.assertEqual(initializer_dependencies.uv_anisotropy_path, actual_crs_to_process[0].uv_anisotropy)
        self.assertEqual(initializer_dependencies.waw_helioion_mp_path, actual_crs_to_process[0].waw_helioion_mp)

        self.assertEqual(expected_l3a_april_paths, actual_crs_to_process[1].l3a_paths)
        self.assertEqual('20100408', actual_crs_to_process[1].cr_midpoint)
        self.assertEqual(2095, actual_crs_to_process[1].cr_rotation_number)
        self.assertEqual(initializer_dependencies.uv_anisotropy_path, actual_crs_to_process[1].uv_anisotropy)
        self.assertEqual(initializer_dependencies.waw_helioion_mp_path, actual_crs_to_process[1].waw_helioion_mp)

        self.assertEqual(Time('2010-01-03 11:33:04.320').value,
                         mock_validate_dependencies.call_args_list[0][0][0].value)
        self.assertEqual(Time('2010-01-31 18:09:30.240').value,
                         mock_validate_dependencies.call_args_list[0][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[0][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[0][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[0][0][4])

        self.assertEqual(Time('2010-02-27 00:45:56.160').value,
                         mock_validate_dependencies.call_args_list[1][0][0].value)
        self.assertEqual(Time('2010-03-27 07:22:22.080').value,
                         mock_validate_dependencies.call_args_list[1][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[1][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[1][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[1][0][4])

        self.assertEqual(Time('2010-03-26 07:22:22.080').value,
                         mock_validate_dependencies.call_args_list[2][0][0].value)
        self.assertEqual(Time('2010-04-23 13:58:48.000').value,
                         mock_validate_dependencies.call_args_list[2][0][1].value)
        self.assertEqual(initializer_dependencies.omni2_data_path,
                         mock_validate_dependencies.call_args_list[2][0][2])
        self.assertEqual(initializer_dependencies.f107_index_file_path,
                         mock_validate_dependencies.call_args_list[2][0][3])
        self.assertEqual(initializer_dependencies.lyman_alpha_path,
                         mock_validate_dependencies.call_args_list[2][0][4])

    @patch("imap_l3_processing.glows.l3b.utils.dump")
    @patch("imap_l3_processing.glows.l3b.utils.ZipFile")
    @patch('builtins.open', new_callable=mock_open, create=True)
    def test_archive_dependencies(self, mocked_open, mock_zip, mock_dump):
        expected_filename = "imap_glows_l3pre-b_l3b-archive_20250314_v001.zip"
        expected_json_filename = "cr_to_process.json"

        dependencies = GlowsInitializerAncillaryDependencies(uv_anisotropy_path="uv_anisotropy",
                                                             lyman_alpha_path=Path("lyman_alpha"),
                                                             omni2_data_path=Path("omni"),
                                                             f107_index_file_path=Path("f107"),
                                                             waw_helioion_mp_path="waw_helioion")

        cr_to_process: CRToProcess = CRToProcess(cr_rotation_number=2095, l3a_paths=[],
                                                 cr_midpoint="20250314",
                                                 waw_helioion_mp=dependencies.waw_helioion_mp_path,
                                                 uv_anisotropy=dependencies.uv_anisotropy_path)

        mock_zip_file = MagicMock()
        mock_zip.return_value.__enter__.return_value = mock_zip_file

        mock_json_file = MagicMock()
        mocked_open.return_value.__enter__.return_value = mock_json_file

        version_number = "v001"
        archive_dependencies(cr_to_process, version_number, dependencies)

        mock_zip.assert_called_with(expected_filename, "w", ZIP_DEFLATED)
        mocked_open.assert_called_once_with(expected_json_filename, "w")

        mock_dump.assert_called_once_with(cr_to_process, mock_json_file)

        mock_zip_file.write.assert_has_calls([
            call(dependencies.lyman_alpha_path),
            call(dependencies.omni2_data_path),
            call(dependencies.f107_index_file_path),
            call(expected_json_filename)
        ])


def create_imap_data_access_json(file_path: str, data_level: str, start_date: str,
                                 descriptor: str = "hist", version: str = "v001") -> dict:
    return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': descriptor,
            'start_date': start_date, 'repointing': None, 'version': version, 'extension': 'pkts',
            'ingestion_date': '2024-10-11 15:28:32'}


def create_l3a_path_by_date(file_date: str) -> str:
    return f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_{file_date}_v001.pkts'
