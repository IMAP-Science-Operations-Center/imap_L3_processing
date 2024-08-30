import os
import shutil
from datetime import date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, EPOCH_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, SwapiL3AlphaSolarWindData


class TestModels(TestCase):
    def setUp(self) -> None:
        imap_processing_folder = Path(imap_processing.__file__).parent

        print(os.getcwd())
        self.temp_directory = imap_processing_folder / "tests" / "test_files"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)

    @patch('imap_processing.swapi.l3a.models.ImapAttributeManager')
    def test_proton_sw_write_cdf(self, mock_imap_attribute_manager_constructor):
        mock_imap_attribute_manager = mock_imap_attribute_manager_constructor.return_value
        mock_imap_attribute_manager.get_global_attributes.return_value = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }
        mock_imap_attribute_manager.get_variable_attributes.side_effect = [
            {
                'epoch1':'epoch_val1',
                'epoch2': 'epoch_val2'
            },
            {
                'proton_sw_speed1': 'proton_sw_speed_val1',
                'proton_sw_speed2': 'proton_sw_speed_val2'
            },
            {
                'proton_sw_speed_unc1': 'proton_sw_speed_unc_val1',
                'proton_sw_speed_unc2': 'proton_sw_speed_unc_val2'
            },
            {
                'epoch_delta1':'epoch_delta_val1',
                'epoch_delta2': 'epoch_delta_val2'
            },
        ]
        epoch = np.arange(10)
        expected_nominal_values = np.full(10, 450)
        expected_std_values = np.full(10, .2)
        speeds = uarray(expected_nominal_values, expected_std_values)
        data = SwapiL3ProtonSolarWindData(epoch, speeds)
        version = "v234"
        file_path = f"{self.temp_directory}/proton_cdf_test.cdf"

        data.write_cdf(file_path, version)

        mock_imap_attribute_manager.add_global_attribute.assert_has_calls(
            [call('Logical_file_id', file_path),
             call('Data_version', version),
             call('Generation_date', date.today().strftime('%Y%m%d'))])


        result_cdf = pycdf.CDF(file_path)
        self.assertEqual('value1',result_cdf.attrs['key1'][...][0])
        self.assertEqual('value2',result_cdf.attrs['key2'][...][0])
        self.assertEqual('value3',result_cdf.attrs['key3'][...][0])
        self.assertEqual(pycdf.const.CDF_TIME_TT2000.value,result_cdf['epoch'].type())
        self.assertEqual('epoch_val1', result_cdf[EPOCH_CDF_VAR_NAME].attrs['epoch1'])
        self.assertEqual('epoch_val2', result_cdf[EPOCH_CDF_VAR_NAME].attrs['epoch2'])
        np.testing.assert_array_equal(epoch, result_cdf.raw_var('epoch')[...])
        self.assertEqual('proton_sw_speed_val1', result_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME].attrs['proton_sw_speed1'])
        self.assertEqual('proton_sw_speed_val2', result_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME].attrs['proton_sw_speed2'])
        np.testing.assert_array_equal(expected_nominal_values, result_cdf[PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME][...])

        self.assertEqual('proton_sw_speed_unc_val1', result_cdf[PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME].attrs['proton_sw_speed_unc1'])
        self.assertEqual('proton_sw_speed_unc_val2', result_cdf[PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME].attrs['proton_sw_speed_unc2'])
        np.testing.assert_array_equal(expected_std_values, result_cdf[PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME][...])
        self.assertEqual('epoch_delta_val1', result_cdf[EPOCH_DELTA_CDF_VAR_NAME].attrs['epoch_delta1'])
        self.assertEqual('epoch_delta_val2', result_cdf[EPOCH_DELTA_CDF_VAR_NAME].attrs['epoch_delta2'])
        self.assertEqual(30000000000,result_cdf[EPOCH_DELTA_CDF_VAR_NAME][...])

    @patch('imap_processing.swapi.l3a.models.ImapAttributeManager')
    def test_alpha_sw_write_cdf(self, mock_imap_attribute_manager_constructor):
        mock_imap_attribute_manager = mock_imap_attribute_manager_constructor.return_value
        mock_imap_attribute_manager.get_global_attributes.return_value = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }
        mock_imap_attribute_manager.get_variable_attributes.side_effect = [
            {
                'epoch1': 'epoch_val1',
                'epoch2': 'epoch_val2'
            },
            {
                'alpha_sw_speed1': 'alpha_sw_speed_val1',
                'alpha_sw_speed2': 'alpha_sw_speed_val2'
            },
            {
                'alpha_sw_speed_unc1': 'alpha_sw_speed_unc_val1',
                'alpha_sw_speed_unc2': 'alpha_sw_speed_unc_val2'
            },
            {
                'epoch_delta1': 'epoch_delta_val1',
                'epoch_delta2': 'epoch_delta_val2'
            },
        ]
        epoch = np.arange(10)
        expected_nominal_values = np.full(10, 450)
        expected_std_values = np.full(10, .2)
        speeds = uarray(expected_nominal_values, expected_std_values)
        data = SwapiL3AlphaSolarWindData(epoch, speeds)
        version = "v234"
        file_path = f"{self.temp_directory}/alpha_cdf_test.cdf"

        data.write_cdf(file_path, version)

        mock_imap_attribute_manager.add_global_attribute.assert_has_calls(
            [call('Logical_file_id', file_path),
             call('Data_version', version),
             call('Generation_date', date.today().strftime('%Y%m%d'))])

        result_cdf = pycdf.CDF(file_path)
        self.assertEqual('value1', result_cdf.attrs['key1'][...][0])
        self.assertEqual('value2', result_cdf.attrs['key2'][...][0])
        self.assertEqual('value3', result_cdf.attrs['key3'][...][0])
        self.assertEqual(pycdf.const.CDF_TIME_TT2000.value, result_cdf['epoch'].type())
        self.assertEqual('epoch_val1', result_cdf[EPOCH_CDF_VAR_NAME].attrs['epoch1'])
        self.assertEqual('epoch_val2', result_cdf[EPOCH_CDF_VAR_NAME].attrs['epoch2'])
        np.testing.assert_array_equal(epoch, result_cdf.raw_var('epoch')[...])
        self.assertEqual('alpha_sw_speed_val1', result_cdf[ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME].attrs['alpha_sw_speed1'])
        self.assertEqual('alpha_sw_speed_val2', result_cdf[ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME].attrs['alpha_sw_speed2'])
        np.testing.assert_array_equal(expected_nominal_values, result_cdf[ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME][...])
        self.assertEqual('alpha_sw_speed_unc_val1',
                         result_cdf[ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME].attrs['alpha_sw_speed_unc1'])
        self.assertEqual('alpha_sw_speed_unc_val2',
                         result_cdf[ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME].attrs['alpha_sw_speed_unc2'])
        np.testing.assert_array_equal(expected_std_values, result_cdf[ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME][...])

        self.assertEqual('epoch_delta_val1', result_cdf[EPOCH_DELTA_CDF_VAR_NAME].attrs['epoch_delta1'])
        self.assertEqual('epoch_delta_val2', result_cdf[EPOCH_DELTA_CDF_VAR_NAME].attrs['epoch_delta2'])
        self.assertEqual(30000000000,result_cdf[EPOCH_DELTA_CDF_VAR_NAME][...])
