import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
from numpy import ndarray
from spacepy import pycdf

from imap_processing.glows.l3a.models import GlowsL3LightCurve, PHOTON_FLUX_CDF_VAR_NAME, EXPOSURE_TIMES_CDF_VAR_NAME, \
    NUM_OF_BINS_CDF_VAR_NAME, BINS_CDF_VAR_NAME, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME, SPIN_ANGLE_CDF_VAR_NAME, LATITUDE_CDF_VAR_NAME, LONGITUDE_CDF_VAR_NAME, \
    EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME, TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME
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

        data = GlowsL3LightCurve(input_metadata=Mock(),
                                 exposure_times=exposure_times,
                                 photon_flux=photon_flux, epoch=epoch,
                                 epoch_delta=epoch_delta, spin_angle=spin_angle,
                                 photon_flux_uncertainty=photon_flux_uncertainty,
                                 latitude=latitudes,
                                 longitude=longitudes,
                                 extra_heliospheric_background=extra_heliospheric_background,
                                 time_dependent_background=time_dependent_background)

        variables = data.to_data_product_variables()
        self.assertEqual(12, len(variables))
        self.assert_variable_attributes(variables[0], photon_flux, PHOTON_FLUX_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[1], photon_flux_uncertainty, PHOTON_FLUX_UNCERTAINTY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[2], exposure_times, EXPOSURE_TIMES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], number_of_bins, NUM_OF_BINS_CDF_VAR_NAME,
                                        expected_record_varying=False, expected_data_type=pycdf.const.CDF_INT2)
        self.assert_variable_attributes(variables[4], np.arange(number_of_bins), BINS_CDF_VAR_NAME,
                                        expected_record_varying=False)
        self.assert_variable_attributes(variables[5], epoch, EPOCH_CDF_VAR_NAME,
                                        expected_data_type=pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(variables[6], epoch_delta, EPOCH_DELTA_CDF_VAR_NAME,
                                        expected_data_type=pycdf.const.CDF_INT8)
        self.assert_variable_attributes(variables[7], spin_angle, SPIN_ANGLE_CDF_VAR_NAME)

        self.assert_variable_attributes(variables[8], latitudes, LATITUDE_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[9], longitudes, LONGITUDE_CDF_VAR_NAME)

        self.assert_variable_attributes(variables[10], extra_heliospheric_background,
                                        EXTRA_HELIOSPHERIC_BACKGROUND_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[11], time_dependent_background,
                                        TIME_DEPENDENT_BACKGROUND_CDF_VAR_NAME)


if __name__ == '__main__':
    unittest.main()
