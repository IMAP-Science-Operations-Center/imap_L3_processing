from unittest.mock import Mock

import numpy as np
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, EPOCH_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME, PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME, \
    SwapiL3AlphaSolarWindData, PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME, ALPHA_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME, ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME, ALPHA_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME, \
    SwapiL3PickupIonData, PUI_COOLING_INDEX_CDF_VAR_NAME, \
    PUI_IONIZATION_RATE_CDF_VAR_NAME, PUI_CUTOFF_SPEED_CDF_VAR_NAME, PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME, \
    PUI_DENSITY_CDF_VAR_NAME, PUI_TEMPERATURE_CDF_VAR_NAME, PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME, \
    PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME, PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME, \
    PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME, PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME, \
    PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, VELOCITY_RTN_LABEL_CDF_VAR_NAME
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):

    def test_getting_proton_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        n = len(epoch_data)

        speed = np.arange(10, step=1.0)
        speed_uncert = np.arange(5, step=.5)
        speed_sun = np.arange(20, 30, step=1.0)
        speed_sun_uncert = np.arange(2, 7, step=.5)
        temperature = np.arange(1000, 2000, step=100.)
        temperature_uncert = np.arange(50, step=5.)
        density = np.arange(3, 13, step=1.)
        density_uncert = np.arange(1, step=.1)
        bulk_v_rtn_sun = np.arange(n * 3, dtype=float).reshape(n, 3)
        bulk_v_rtn_sc = np.arange(100, 100 + n * 3, dtype=float).reshape(n, 3)
        bulk_v_rtn_cov = np.arange(200, 200 + n * 9, dtype=float).reshape(n, 3, 3)

        quality_flags = np.full(n, SwapiL3Flags.NONE)
        quality_flags[3:5] |= SwapiL3Flags.FIT_ERROR

        data = SwapiL3ProtonSolarWindData(
            Mock(), epoch_data,
            speed, speed_uncert,
            speed_sun, speed_sun_uncert,
            temperature, temperature_uncert,
            density, density_uncert,
            bulk_v_rtn_sun, bulk_v_rtn_sc, bulk_v_rtn_cov,
            quality_flags,
        )

        variables = data.to_data_product_variables()

        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[1], speed, PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[2], speed_uncert, PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], speed_sun, PROTON_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[4], speed_sun_uncert,
                                        PROTON_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[5], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[6], temperature, PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[7], temperature_uncert,
                                        PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[8], density, PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[9], density_uncert,
                                        PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[10], bulk_v_rtn_sun,
                                        PROTON_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[11], bulk_v_rtn_sc,
                                        PROTON_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[12], bulk_v_rtn_cov,
                                        PROTON_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME)
        expected_velocity_uncert = np.sqrt(np.diagonal(bulk_v_rtn_cov, axis1=1, axis2=2))
        self.assert_variable_attributes(variables[13], expected_velocity_uncert,
                                        PROTON_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[14], quality_flags, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[15], ["R", "T", "N"], VELOCITY_RTN_LABEL_CDF_VAR_NAME)

    def test_getting_alpha_sw_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        epoch_delta = np.full_like(epoch_data, THIRTY_SECONDS_IN_NANOSECONDS)
        n = len(epoch_data)

        speed = np.arange(400, 400 + n, dtype=float)
        speed_uncert = np.arange(1, 1 + n, dtype=float)
        speed_sun = np.arange(420, 420 + n, dtype=float)
        speed_sun_uncert = np.arange(2, 2 + n, dtype=float)
        density = np.arange(2, step=.2)
        density_uncert = np.arange(1, step=0.1)
        temperature = np.arange(300000, step=30000.)
        temperature_uncert = np.arange(50000, step=5000.)
        velocity_rtn_sun = np.arange(n * 3, dtype=float).reshape(n, 3)
        velocity_rtn_sc = np.arange(100, 100 + n * 3, dtype=float).reshape(n, 3)
        velocity_rtn_cov = np.arange(n * 9, dtype=float).reshape(n, 3, 3)

        quality_flags = np.full_like(epoch_data, SwapiL3Flags.NONE)
        quality_flags[:n // 2] = SwapiL3Flags.BAD_FIT

        data = SwapiL3AlphaSolarWindData(
            Mock(), epoch_data,
            speed, speed_uncert,
            speed_sun, speed_sun_uncert,
            density, density_uncert,
            temperature, temperature_uncert,
            velocity_rtn_sun, velocity_rtn_sc, velocity_rtn_cov,
            quality_flags,
        )
        variables = data.to_data_product_variables()

        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[1], speed, ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[2], speed_uncert,
                                        ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], speed_sun, ALPHA_SOLAR_WIND_SPEED_SUN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[4], speed_sun_uncert,
                                        ALPHA_SOLAR_WIND_SPEED_SUN_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[5], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[6], density, ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[7], density_uncert,
                                        ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[8], temperature, ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[9], temperature_uncert,
                                        ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[10], velocity_rtn_sun,
                                        ALPHA_SOLAR_WIND_VELOCITY_RTN_SUN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[11], velocity_rtn_sc,
                                        ALPHA_SOLAR_WIND_VELOCITY_RTN_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[12], velocity_rtn_cov,
                                        ALPHA_SOLAR_WIND_VELOCITY_RTN_COVARIANCE_CDF_VAR_NAME)
        expected_velocity_uncert = np.sqrt(np.diagonal(velocity_rtn_cov, axis1=1, axis2=2))
        self.assert_variable_attributes(variables[13], expected_velocity_uncert,
                                        ALPHA_SOLAR_WIND_VELOCITY_RTN_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[14], quality_flags, SWAPI_QUALITY_FLAGS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[15], ["R", "T", "N"], VELOCITY_RTN_LABEL_CDF_VAR_NAME)

    def test_getting_pui_data_product_variables(self):
        epoch_data = np.arange(20, step=2)
        expected_epoch_delta = np.full(10, FIVE_MINUTES_IN_NANOSECONDS)
        expected_cooling_index_nominal = np.arange(10, step=1.)
        expected_cooling_index_std_dev = np.arange(.1, step=.01)
        expected_cooling_index = uarray(expected_cooling_index_nominal, expected_cooling_index_std_dev)

        expected_ionization_rate_nominal = np.arange(300000, step=30000.)
        expected_ionization_rate_std_dev = np.arange(10, step=1.)
        expected_ionization_rate = uarray(expected_ionization_rate_nominal, expected_ionization_rate_std_dev)

        expected_cutoff_speed_nominal = np.arange(50000, step=5000.)
        expected_cutoff_speed_std_dev = np.arange(5, step=.5)
        expected_cutoff_speed = uarray(expected_cutoff_speed_nominal, expected_cutoff_speed_std_dev)

        expected_background_nominal = np.arange(1, step=0.1)
        expected_background_std_dev = np.arange(.01, step=0.001)
        expected_background = uarray(expected_background_nominal, expected_background_std_dev)

        expected_density_nominal = np.arange(0.001, step=0.0001)
        expected_density_std_dev = np.arange(0.0001, step=0.00001)
        expected_density = uarray(expected_density_nominal, expected_density_std_dev)

        expected_temperature_nominal = np.arange(1e5, step=10000.)
        expected_temperature_std_dev = np.arange(100, step=10.)
        expected_temperature = uarray(expected_temperature_nominal, expected_temperature_std_dev)

        expected_quality_flags = np.full(20, 0)

        data = SwapiL3PickupIonData(Mock(), epoch_data, expected_cooling_index, expected_ionization_rate,
                                    expected_cutoff_speed, expected_background, expected_density, expected_temperature,
                                    expected_quality_flags)
        variables = data.to_data_product_variables()

        self.assertEqual(15, len(variables))
        self.assert_variable_attributes(variables[0], epoch_data, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[1], expected_epoch_delta, EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[2], expected_cooling_index_nominal,
                                        PUI_COOLING_INDEX_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], expected_cooling_index_std_dev,
                                        PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[4], expected_ionization_rate_nominal,
                                        PUI_IONIZATION_RATE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[5], expected_ionization_rate_std_dev,
                                        PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[6], expected_cutoff_speed_nominal,
                                        PUI_CUTOFF_SPEED_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[7], expected_cutoff_speed_std_dev,
                                        PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[8], expected_background_nominal,
                                        PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[9], expected_background_std_dev,
                                        PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[10], expected_density_nominal,
                                        PUI_DENSITY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[11], expected_density_std_dev,
                                        PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[12], expected_temperature_nominal,
                                        PUI_TEMPERATURE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[13], expected_temperature_std_dev,
                                        PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[14], expected_quality_flags,
                                        SWAPI_QUALITY_FLAGS_CDF_VAR_NAME)
