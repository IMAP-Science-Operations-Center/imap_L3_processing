from dataclasses import dataclass

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import nominal_values, std_devs

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable
from imap_l3_processing.swapi.l3a.models import EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME

SOLAR_WIND_ENERGY_CDF_VAR_NAME = "combined_energy"
SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME = "combined_energy_delta_minus"
SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME = "combined_energy_delta_plus"

COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME = "combined_differential_flux"
COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME = "combined_differential_flux_delta"


@dataclass
class SwapiL3BCombinedVDF(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_velocities: np.ndarray[float]
    proton_sw_velocities_delta_minus: np.ndarray[float]
    proton_sw_velocities_delta_plus: np.ndarray[float]
    proton_sw_combined_vdf: np.ndarray[float]
    alpha_sw_velocities: np.ndarray[float]
    alpha_sw_velocities_delta_minus: np.ndarray[float]
    alpha_sw_velocities_delta_plus: np.ndarray[float]
    alpha_sw_combined_vdf: np.ndarray[float]
    pui_sw_velocities: np.ndarray[float]
    pui_sw_velocities_delta_minus: np.ndarray[float]
    pui_sw_velocities_delta_plus: np.ndarray[float]
    pui_sw_combined_vdf: np.ndarray[float]
    combined_energy: np.ndarray[float]
    combined_energy_delta_minus: np.ndarray[float]
    combined_energy_delta_plus: np.ndarray[float]
    combined_differential_flux: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, FIVE_MINUTES_IN_NANOSECONDS)),

            DataProductVariable(SOLAR_WIND_ENERGY_CDF_VAR_NAME, self.combined_energy),
            DataProductVariable(SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.combined_energy_delta_minus),
            DataProductVariable(SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.combined_energy_delta_plus),

            DataProductVariable(COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME,
                                nominal_values(self.combined_differential_flux)),
            DataProductVariable(COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME,
                                std_devs(self.combined_differential_flux))
        ]
