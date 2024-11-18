import unittest
from unittest.mock import Mock

import numpy as np
from numpy import ndarray

from imap_processing.glows.l3a.models import GlowsL3LightCurve, PHOTON_FLUX_CDF_VAR_NAME, EXPOSURE_TIMES_CDF_VAR_NAME, \
    NUM_OF_BINS_CDF_VAR_NAME, BINS_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_data_to_product_variables(self):
        photon_flux: ndarray = np.arange(360)+50
        exposure_times: ndarray = np.arange(360) + 100
        number_of_bins: int = 360

        data = GlowsL3LightCurve(input_metadata=Mock(),
                          exposure_times=exposure_times,
                          photon_flux=photon_flux)

        variables = data.to_data_product_variables()
        self.assertEqual(4, len(variables))
        self.assert_variable_attributes(variables[0], photon_flux, PHOTON_FLUX_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[1], exposure_times, EXPOSURE_TIMES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[2], number_of_bins, NUM_OF_BINS_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(variables[3], np.arange(number_of_bins), BINS_CDF_VAR_NAME)


if __name__ == '__main__':
    unittest.main()
