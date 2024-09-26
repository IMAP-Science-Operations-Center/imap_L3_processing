from datetime import datetime

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import uarray

from imap_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.models import EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME
from imap_processing.swapi.l3b.models import SwapiL3BCombinedVDF, PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME, PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_combined_vdf_data_products(self):
        input_metadata = UpstreamDataDependency("swapi", "l3b",
                                                datetime(2024, 9, 8),
                                                datetime(2024, 9, 9),
                                                "v001", "")
        epoch = np.array([1, 2, 3])
        proton_velocities = np.array([4, 5, 6])
        proton_vdf = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
        proton_vdf_uncertainties = np.array([[0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]])

        vdf = SwapiL3BCombinedVDF(input_metadata, epoch, proton_velocities,
                                  uarray(proton_vdf, proton_vdf_uncertainties))
        variables = vdf.to_data_product_variables()

        self.assertEqual(6, len(variables))
        self.assert_variable_attributes(variables[0], epoch, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(variables[1], FIVE_MINUTES_IN_NANOSECONDS, EPOCH_DELTA_CDF_VAR_NAME,
                                        expected_record_varying=False)
        self.assert_variable_attributes(variables[2], proton_velocities, PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], 6.0, PROTON_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME,
                                        expected_record_varying=False)
        self.assert_variable_attributes(variables[4], proton_vdf, PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[5], proton_vdf_uncertainties,
                                        PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME)
