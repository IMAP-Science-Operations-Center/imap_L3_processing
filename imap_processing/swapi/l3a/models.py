from dataclasses import dataclass

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import nominal_values, std_devs

from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS
from imap_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_delta"
PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "proton_sw_temperature"
PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_temperature_delta"
PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "proton_sw_density"
PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_density_delta"
ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME = "alpha_sw_speed"
ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_delta"


@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]
    proton_sw_temperature: np.ndarray[float]
    proton_sw_density: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.proton_sw_speed)),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.proton_sw_speed)),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, THIRTY_SECONDS_IN_NANOSECONDS, record_varying=False),
            DataProductVariable(PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.proton_sw_temperature)),
            DataProductVariable(PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_temperature)),
            DataProductVariable(PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME, nominal_values(self.proton_sw_density)),
            DataProductVariable(PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_density)),
        ]


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    alpha_sw_speed: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.alpha_sw_speed)),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.alpha_sw_speed)),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, THIRTY_SECONDS_IN_NANOSECONDS, record_varying=False)
        ]


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    spin_angles: np.ndarray[float]  # not currently in the L2 cdf, is in the sample data provided by Bishwas
    coincidence_count_rate_uncertainty: np.ndarray[float]
