import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, sentinel, call

import numpy as np
from spacepy import pycdf
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

import imap_processing
from imap_processing.constants import TEMP_CDF_FOLDER_PATH, THIRTY_SECONDS_IN_NANOSECONDS
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.processor import SwapiL3AProcessor, EPOCH_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME


class TestProcessor(TestCase):
    def setUp(self) -> None:
        self.temp_directory = f"{TEMP_CDF_FOLDER_PATH}"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

        self.mock_imap_patcher = patch('imap_processing.swapi.l3a.processor.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
        self.mock_imap_patcher.stop()


    @patch('imap_processing.swapi.l3a.processor.uuid')
    @patch('imap_processing.swapi.l3a.processor.chunk_l2_data')
    @patch('imap_processing.swapi.l3a.processor.read_l2_swapi_data')
    @patch('imap_processing.swapi.l3a.processor.calculate_proton_solar_wind_speed')
    @patch('imap_processing.swapi.l3a.processor.calculate_alpha_solar_wind_speed')
    def test_processor(self, mock_calculate_alpha_solar_wind_speed, mock_calculate_proton_solar_wind_speed, mock_read_l2_swapi_data, mock_chunk_l2_data, mock_uuid):

        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        mock_uuid_value = 123
        mock_uuid.uuid4.return_value = mock_uuid_value

        self.mock_imap_api.query.return_value = [{'file_path':sentinel.file_path}]
        self.mock_imap_api.download.return_value = file_path
        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = 'c'
        end_date = datetime.now()
        version = 'f'
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

        actual_proton_sw_speed = ufloat(400000, 2)
        mock_calculate_proton_solar_wind_speed.return_value = (actual_proton_sw_speed, sentinel.a, sentinel.phi, sentinel.b)

        mock_calculate_alpha_solar_wind_speed.return_value = ufloat(450000, 1000)

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        spin_angles = np.array([[24, 25, 26, 27, 28], [29, 30, 31, 32, 33], [34, 35, 36, 37, 38], [39, 40, 41, 42, 43]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_chunk_l2_data.return_value = [SwapiL2Data(epoch,energy,coincidence_count_rate,spin_angles,coincidence_count_rate_uncertainty)]

        swapi_processor = SwapiL3AProcessor([UpstreamDataDependency(instrument, incoming_data_level, descriptor, start_date, end_date,
                                                                    version)], instrument, outgoing_data_level, start_date, end_date,
                                            outgoing_version)
        swapi_processor.process()

        start_date_as_str = start_date.strftime("%Y%d%m")
        end_date_as_str = end_date.strftime("%Y%d%m")
        self.mock_imap_api.query.assert_called_with(instrument=instrument, data_level=incoming_data_level, descriptor=descriptor, start_date=start_date_as_str, end_date=end_date_as_str, version='latest')
        self.mock_imap_api.download.assert_called_with(sentinel.file_path)
        mock_chunk_l2_data.assert_called_with(mock_read_l2_swapi_data.return_value, 5)
        expected_count_rate_with_uncertainties = uarray(coincidence_count_rate, coincidence_count_rate_uncertainty)
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties), nominal_values(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(spin_angles,mock_calculate_proton_solar_wind_speed.call_args_list[0].args[1])
        np.testing.assert_array_equal(energy,mock_calculate_proton_solar_wind_speed.call_args_list[0].args[2])
        np.testing.assert_array_equal(epoch,mock_calculate_proton_solar_wind_speed.call_args_list[0].args[3])

        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(energy,mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[1])


        proton_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_proton-solar-wind-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        result_cdf = pycdf.CDF(proton_cdf_path)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), result_cdf[EPOCH_CDF_VAR_NAME])
        np.testing.assert_array_equal(np.array([400000]), result_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME])
        np.testing.assert_array_equal(np.array([2]), result_cdf[PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME])
        self.assertEqual(THIRTY_SECONDS_IN_NANOSECONDS,result_cdf[EPOCH_DELTA_CDF_VAR_NAME][...])
        self.assertFalse(result_cdf[EPOCH_DELTA_CDF_VAR_NAME].rv())

        alpha_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_alpha-solar-wind-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        result_cdf = pycdf.CDF(alpha_cdf_path)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), result_cdf[EPOCH_CDF_VAR_NAME])
        np.testing.assert_array_equal(np.array([450000]), result_cdf[ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME])
        np.testing.assert_array_equal(np.array([1000]), result_cdf[ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME])
        self.assertEqual(THIRTY_SECONDS_IN_NANOSECONDS, result_cdf[EPOCH_DELTA_CDF_VAR_NAME][...])
        self.assertFalse(result_cdf[EPOCH_DELTA_CDF_VAR_NAME].rv())
        self.mock_imap_api.upload.assert_has_calls([call(proton_cdf_path), call(alpha_cdf_path)])


    def test_processor_throws_exception_when_more_than_one_file_is_downloaded(self):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        self.mock_imap_api.query.return_value = [{'file_path': sentinel.file_path}]
        self.mock_imap_api.download.return_value = file_path

        self.mock_imap_api.query.return_value = [{'file_path':'1 thing'}, {'file_path':'2 thing'}]
        swapi_processor = SwapiL3AProcessor(
            [UpstreamDataDependency('swapi', 'l2', 'c', datetime.now() - timedelta(days=1), datetime.now(),
                                    'f')], 'swapi', "l3a", datetime.now() - timedelta(days=1),
            datetime.now(),
            "12345")

        try:
            swapi_processor.process()
            self.fail()
        except ValueError as e:
            return

