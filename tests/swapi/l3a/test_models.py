from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import uarray

from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, EPOCH_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    DataProductVariable, SwapiL3AlphaSolarWindData, ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME


class TestModels(TestCase):
    def test_getting_proton_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        expected_nominal_values = np.arange(10, step=1)
        expected_std = np.arange(5, step=.5)
        proton_speed = uarray(expected_nominal_values, expected_std)
        expected_temperature_nominal_values = np.arange(1000, 2000, step=100)
        expected_temperature_std = np.arange(50, step=5)
        temperature_data = uarray(expected_temperature_nominal_values, expected_temperature_std)
        expected_density_nominal_values = np.arange(3, 13, step=1)
        expected_density_std = np.arange(1, step=.1)
        density_data = uarray(expected_density_nominal_values, expected_density_std)
        data = SwapiL3ProtonSolarWindData(Mock(), epoch_data, proton_speed, temperature_data, density_data, Mock(),
                                          Mock())
        variables = data.to_data_product_variables()

        self._assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self._assert_variable_attributes(variables[1], expected_nominal_values, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[2], expected_std, PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[3], THIRTY_SECONDS_IN_NANOSECONDS, EPOCH_DELTA_CDF_VAR_NAME,
                                         expected_record_varying=False)
        self._assert_variable_attributes(variables[4], expected_temperature_nominal_values,
                                         PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[5], expected_temperature_std,
                                         PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[6], expected_density_nominal_values,
                                         PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[7], expected_density_std,
                                         PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME)

    def test_getting_alpha_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        expected_nominal_values = np.arange(10, step=1)
        expected_std = np.arange(5, step=.5)
        proton_speed = uarray(expected_nominal_values, expected_std)
        data = SwapiL3AlphaSolarWindData(Mock(), epoch_data, proton_speed)
        variables = data.to_data_product_variables()

        self._assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self._assert_variable_attributes(variables[1], expected_nominal_values, ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[2], expected_std, ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME)
        self._assert_variable_attributes(variables[3], THIRTY_SECONDS_IN_NANOSECONDS, EPOCH_DELTA_CDF_VAR_NAME,
                                         expected_record_varying=False)

    def _assert_variable_attributes(self, variable: DataProductVariable,
                                    expected_data, expected_name,
                                    expected_data_type=None,
                                    expected_record_varying=True):
        self.assertEqual(expected_name, variable.name)
        np.testing.assert_array_equal(expected_data, variable.value)
        self.assertEqual(expected_data_type, variable.cdf_data_type)
        self.assertEqual(expected_record_varying, variable.record_varying)
