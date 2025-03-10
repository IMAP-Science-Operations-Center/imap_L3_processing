from datetime import datetime

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swapi.l3a.models import EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF, PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME, PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, PUI_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, \
    PUI_SOLAR_WIND_VDF_CDF_VAR_NAME, \
    PUI_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME, \
    COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME, SOLAR_WIND_ENERGY_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME, \
    PROTON_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME, ALPHA_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME, \
    ALPHA_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME, PUI_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME, \
    PUI_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME, SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME, \
    SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_combined_vdf_data_products(self):
        input_metadata = UpstreamDataDependency("swapi", "l3b",
                                                datetime(2024, 9, 8),
                                                datetime(2024, 9, 9),
                                                "v001", "")
        epoch = np.array([1, 2, 3])
        proton_velocities = np.array([4, 5, 6])
        proton_velocities_delta_plus = np.array([0.4, 0.5, 0.6])
        proton_velocities_delta_minus = 1 + np.array([0.4, 0.5, 0.6])
        proton_vdf = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
        proton_vdf_uncertainties = np.array([[0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]])
        alpha_velocities = np.array([11, 12, 13])
        alpha_velocities_delta_plus = np.array([0.11, 0.12, 0.13])
        alpha_velocities_delta_minus = 1 + np.array([0.11, 0.12, 0.13])
        alpha_vdf = np.array([[14, 15, 16], [17, 18, 19], [20, 21, 22]])
        alpha_vdf_uncertainties = np.array([[0.7, 0.8, 0.9], [0.10, 0.11, 0.12], [0.13, 0.14, 0.15]])
        pui_velocities = np.array([23, 24, 25])
        pui_velocities_delta_plus = np.array([0.23, 0.24, 0.25])
        pui_velocities_delta_minus = 1 + np.array([0.23, 0.24, 0.25])
        pui_vdf = np.array([[26, 27, 28], [29, 30, 31], [32, 33, 34]])
        pui_vdf_uncertainties = np.array([[0.72, 0.8, 0.9], [0.10, 0.121, 0.12], [0.13, 0.142, 0.15]])
        combined_energies = np.array([230, 240, 250])
        combined_energies_delta_plus = np.array([44, 55, 66])
        combined_energies_delta_minus = 1 + np.array([44, 55, 66])
        combined_differential_flux = np.array([[26, 27.2, 28], [29.2, 30, 31], [32, 33.5, 34]])
        combined_differential_flux_uncertainties = np.array(
            [[0.725, 0.8, 0.9], [0.105, 0.121, 0.124], [0.13, 0.1425, 0.15]])

        vdf = SwapiL3BCombinedVDF(
            input_metadata=input_metadata,
            epoch=epoch,
            proton_sw_velocities=proton_velocities,
            proton_sw_velocities_delta_minus=proton_velocities_delta_minus,
            proton_sw_velocities_delta_plus=proton_velocities_delta_plus,
            proton_sw_combined_vdf=uarray(proton_vdf, proton_vdf_uncertainties),
            alpha_sw_velocities=alpha_velocities,
            alpha_sw_velocities_delta_minus=alpha_velocities_delta_minus,
            alpha_sw_velocities_delta_plus=alpha_velocities_delta_plus,
            alpha_sw_combined_vdf=uarray(alpha_vdf, alpha_vdf_uncertainties),
            pui_sw_velocities=pui_velocities,
            pui_sw_velocities_delta_minus=pui_velocities_delta_minus,
            pui_sw_velocities_delta_plus=pui_velocities_delta_plus,
            pui_sw_combined_vdf=uarray(pui_vdf, pui_vdf_uncertainties),
            combined_energy=combined_energies,
            combined_energy_delta_minus=combined_energies_delta_minus,
            combined_energy_delta_plus=combined_energies_delta_plus,
            combined_differential_flux=uarray(combined_differential_flux, combined_differential_flux_uncertainties))

        variables = vdf.to_data_product_variables()

        self.assertEqual(22, len(variables))
        self.assert_variable_attributes(variables[0], epoch, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(variables[1], FIVE_MINUTES_IN_NANOSECONDS, EPOCH_DELTA_CDF_VAR_NAME,
                                        expected_record_varying=False)
        self.assert_variable_attributes(variables[2], proton_velocities, PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[3], proton_velocities_delta_minus,
                                        PROTON_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[4], proton_velocities_delta_plus,
                                        PROTON_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[5], proton_vdf, PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[6], proton_vdf_uncertainties,
                                        PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[7], alpha_velocities, ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[8], alpha_velocities_delta_minus,
                                        ALPHA_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[9], alpha_velocities_delta_plus,
                                        ALPHA_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[10], alpha_vdf, ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[11], alpha_vdf_uncertainties,
                                        ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME)

        self.assert_variable_attributes(variables[12], pui_velocities, PUI_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[13], pui_velocities_delta_minus,
                                        PUI_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[14], pui_velocities_delta_plus,
                                        PUI_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[15], pui_vdf, PUI_SOLAR_WIND_VDF_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[16], pui_vdf_uncertainties, PUI_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME)

        self.assert_variable_attributes(variables[17], combined_energies, SOLAR_WIND_ENERGY_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[18], combined_energies_delta_minus,
                                        SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[19], combined_energies_delta_plus,
                                        SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[20], combined_differential_flux,
                                        COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME)
        self.assert_variable_attributes(variables[21], combined_differential_flux_uncertainties,
                                        COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME)
