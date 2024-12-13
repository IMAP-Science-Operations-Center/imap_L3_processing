import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
from numpy import ndarray
from spacepy import pycdf

from imap_processing.glows.l3a.models import GlowsL3LightCurve, PHOTON_FLUX_CDF_VAR_NAME, EXPOSURE_TIMES_CDF_VAR_NAME, \
    NUM_OF_BINS_CDF_VAR_NAME, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME, SPIN_ANGLE_CDF_VAR_NAME, LATITUDE_CDF_VAR_NAME, LONGITUDE_CDF_VAR_NAME, \
    EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME, TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME, \
    FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME, FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME, HV_VOLTAGE_AVERAGE_CDF_VAR_NAME, \
    HV_VOLTAGE_STD_DEV_CDF_VAR_NAME, SPIN_PERIOD_AVERAGE_CDF_VAR_NAME, SPIN_PERIOD_STD_DEV_CDF_VAR_NAME, \
    SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME, SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME, PULSE_LENGTH_AVERAGE_CDF_VAR_NAME, \
    PULSE_LENGTH_STD_DEV_CDF_VAR_NAME, POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME, \
    POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME, SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME, \
    SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME, SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME, \
    SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME, SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME, \
    SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_data_to_product_variables(self):
        photon_flux: ndarray = (np.arange(360) + 50).reshape(1, -1)
        photon_flux_uncertainty = photon_flux / 100.
        exposure_times: ndarray = (np.arange(360) + 100).reshape(1, -1)
        epoch: ndarray = np.array(datetime(2024, 11, 18, 12))
        epoch_delta = np.array(43200000000000)
        spin_angle = photon_flux + 1.8
        latitudes: ndarray = (np.arange(360) + 360).reshape(1, -1)
        longitudes: ndarray = (np.arange(360) + 180).reshape(1, -1)
        extra_heliospheric_background: ndarray = (np.arange(360) + 25).reshape(1, -1)
        time_dependent_background: ndarray = (np.arange(360) + 10).reshape(1, -1)
        number_of_bins: int = 360

        filter_temperature_average = Mock()
        filter_temperature_std_dev = Mock()
        hv_voltage_average = Mock()
        hv_voltage_std_dev = Mock()
        spin_period_average = Mock()
        spin_period_std_dev = Mock()
        spin_period_ground_average = Mock()
        spin_period_ground_std_dev = Mock()
        pulse_length_average = Mock()
        pulse_length_std_dev = Mock()
        position_angle_offset_average = Mock()
        position_angle_offset_std_dev = Mock()

        spin_axis_orientation_average = Mock()
        spin_axis_orientation_std_dev = Mock()
        spacecraft_location_average = Mock()
        spacecraft_location_std_dev = Mock()
        spacecraft_velocity_average = Mock()
        spacecraft_velocity_std_dev = Mock()

        data = GlowsL3LightCurve(input_metadata=Mock(),
                                 exposure_times=exposure_times,
                                 photon_flux=photon_flux, epoch=epoch,
                                 epoch_delta=epoch_delta, spin_angle=spin_angle,
                                 photon_flux_uncertainty=photon_flux_uncertainty,
                                 latitude=latitudes,
                                 longitude=longitudes,
                                 extra_heliospheric_background=extra_heliospheric_background,
                                 time_dependent_background=time_dependent_background,
                                 filter_temperature_average=filter_temperature_average,
                                 filter_temperature_std_dev=filter_temperature_std_dev,
                                 hv_voltage_average=hv_voltage_average,
                                 hv_voltage_std_dev=hv_voltage_std_dev,
                                 spin_period_average=spin_period_average,
                                 spin_period_std_dev=spin_period_std_dev,
                                 spin_period_ground_average=spin_period_ground_average,
                                 spin_period_ground_std_dev=spin_period_ground_std_dev,
                                 pulse_length_average=pulse_length_average,
                                 pulse_length_std_dev=pulse_length_std_dev,
                                 position_angle_offset_average=position_angle_offset_average,
                                 position_angle_offset_std_dev=position_angle_offset_std_dev,
                                 spin_axis_orientation_average=spin_axis_orientation_average,
                                 spin_axis_orientation_std_dev=spin_axis_orientation_std_dev,
                                 spacecraft_location_average=spacecraft_location_average,
                                 spacecraft_location_std_dev=spacecraft_location_std_dev,
                                 spacecraft_velocity_average=spacecraft_velocity_average,
                                 spacecraft_velocity_std_dev=spacecraft_velocity_std_dev, )

        variables = data.to_data_product_variables()
        self.assertEqual(29, len(variables))
        variables = iter(variables)

        # @formatter:off
        self.assert_variable_attributes(next(variables), photon_flux, PHOTON_FLUX_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), photon_flux_uncertainty, PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), exposure_times, EXPOSURE_TIMES_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), number_of_bins, NUM_OF_BINS_CDF_VAR_NAME, expected_record_varying=False, expected_data_type=pycdf.const.CDF_INT2)
        self.assert_variable_attributes(next(variables), epoch, EPOCH_CDF_VAR_NAME, expected_data_type=pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(next(variables), epoch_delta, EPOCH_DELTA_CDF_VAR_NAME, expected_data_type=pycdf.const.CDF_INT8)
        self.assert_variable_attributes(next(variables), spin_angle, SPIN_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), latitudes, LATITUDE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), longitudes, LONGITUDE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), extra_heliospheric_background, EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), time_dependent_background, TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), filter_temperature_average, FILTER_TEMPERATURE_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), filter_temperature_std_dev, FILTER_TEMPERATURE_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), hv_voltage_average, HV_VOLTAGE_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), hv_voltage_std_dev, HV_VOLTAGE_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_period_average, SPIN_PERIOD_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_period_std_dev, SPIN_PERIOD_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_period_ground_average, SPIN_PERIOD_GROUND_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_period_ground_std_dev, SPIN_PERIOD_GROUND_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), pulse_length_average, PULSE_LENGTH_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), pulse_length_std_dev, PULSE_LENGTH_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), position_angle_offset_average, POSITION_ANGLE_OFFSET_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), position_angle_offset_std_dev, POSITION_ANGLE_OFFSET_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_axis_orientation_average, SPIN_AXIS_ORIENTATION_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spin_axis_orientation_std_dev, SPIN_AXIS_ORIENTATION_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spacecraft_location_average, SPACECRAFT_LOCATION_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spacecraft_location_std_dev, SPACECRAFT_LOCATION_STD_DEV_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spacecraft_velocity_average, SPACECRAFT_VELOCITY_AVERAGE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), spacecraft_velocity_std_dev, SPACECRAFT_VELOCITY_STD_DEV_CDF_VAR_NAME)


if __name__ == '__main__':
    unittest.main()
