import unittest
from datetime import datetime, timedelta

from astropy.time import Time
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3b.models import CRToProcess
from imap_l3_processing.glows.l3b.utils import read_glows_l3a_data, find_unprocessed_carrington_rotations
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
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle[0][0])
        self.assertEqual(2.0, actual_glows_lightcurve.spin_angle_delta[0][0])
        self.assertEqual(162.0919952392578, actual_glows_lightcurve.spin_axis_orientation_average[0][0])
        self.assertEqual(0.2345000058412552, actual_glows_lightcurve.spin_axis_orientation_std_dev[0][0])
        self.assertEqual(15.0, actual_glows_lightcurve.spin_period_average[0])
        self.assertEqual(15.236681938171387, actual_glows_lightcurve.spin_period_ground_average[0])
        self.assertEqual(0.0014979999978095293, actual_glows_lightcurve.spin_period_ground_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.spin_period_std_dev[0])
        self.assertEqual(0.0, actual_glows_lightcurve.time_dependent_background[0][0])

    def test_find_unprocessed_carrington_rotations(self):
        l3a_files_january = [
            self.create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201001{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201001{str(i).zfill(2)}') for i in range(1, 32)
        ]
        l3a_files_february = [
            self.create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201002{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201002{str(i).zfill(2)}') for i in range(1, 29)
        ]
        l3a_files_march = [
            self.create_imap_data_access_json(
                file_path=f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_201003{str(i).zfill(2)}_v001.pkts',
                data_level='l3a', start_date=f'201003{str(i).zfill(2)}') for i in range(1, 32)
        ]

        l3a_files = l3a_files_february + l3a_files_march + l3a_files_january

        l3b_files = [
            self.create_imap_data_access_json(
                file_path=f'imap/glows/l3b/2010/01/imap_glows_l3b_hist_20100213_v001.pkts',
                data_level='l3b', start_date=f'20100213')
        ]

        expected_l3a_january_paths = [self.create_l3a_path_by_date(f'201001{str(i).zfill(2)}') for i in range(3, 31)]
        expected_l3a_march_paths = [self.create_l3a_path_by_date(f'20100227'),
                                    self.create_l3a_path_by_date(f'20100228')] + [
                                       self.create_l3a_path_by_date(f'201003{str(i).zfill(2)}') for i in range(1, 27)]

        actual_crs_to_process: [CRToProcess] = find_unprocessed_carrington_rotations(l3a_files, l3b_files)

        self.assertEqual(2, len(actual_crs_to_process))
        self.assertEqual(expected_l3a_january_paths, actual_crs_to_process[0].l3a_paths)
        self.assertEqual(Time('2010-01-03 11:33:04.320').value, actual_crs_to_process[0].carrington_start_date.value)
        self.assertEqual(Time('2010-01-30 18:09:30.240').value, actual_crs_to_process[0].carrington_end_date.value)
        self.assertEqual(2092, actual_crs_to_process[0].carrington_rotation)

        self.assertEqual(expected_l3a_march_paths, actual_crs_to_process[1].l3a_paths)
        self.assertEqual(Time('2010-02-27 00:45:56.160').value, actual_crs_to_process[1].carrington_start_date.value)
        self.assertEqual(Time('2010-03-26 07:22:22.080').value, actual_crs_to_process[1].carrington_end_date.value)
        self.assertEqual(2094, actual_crs_to_process[1].carrington_rotation)

    def create_imap_data_access_json(self, file_path: str, data_level: str, start_date: str) -> dict:
        return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': 'hist',
                'start_date': start_date, 'repointing': None, 'version': 'v001', 'extension': 'pkts',
                'ingestion_date': '2024-10-11 15:28:32'}

    def create_l3a_path_by_date(self, file_date: str) -> str:
        return f'imap/glows/l3a/2010/01/imap_glows_l3a_hist_{file_date}_v001.pkts'
