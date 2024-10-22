from dataclasses import dataclass

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import nominal_values, std_devs

from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_delta"
PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "proton_sw_temperature"
PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_temperature_delta"
PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "proton_sw_density"
PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_density_delta"

PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME = "proton_sw_clock_angle"
PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_clock_angle_delta"

PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME = "proton_sw_deflection_angle"
PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_deflection_angle_delta"

ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME = "alpha_sw_speed"
ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_delta"
ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "alpha_sw_density"
ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_density_delta"
ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_temperature"
ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_temperature_delta"

PUI_COOLING_INDEX_CDF_VAR_NAME = "pui_cooling_index"
PUI_IONIZATION_RATE_CDF_VAR_NAME = "pui_ionization_rate"
PUI_CUTOFF_SPEED_CDF_VAR_NAME = "pui_cutoff_speed"
PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME = "pui_background_count_rate"


@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]
    proton_sw_temperature: np.ndarray[float]
    proton_sw_density: np.ndarray[float]
    proton_sw_clock_angle: np.ndarray[float]
    proton_sw_deflection_angle: np.ndarray[float]

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
            DataProductVariable(PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME, nominal_values(self.proton_sw_clock_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_clock_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME,
                                nominal_values(self.proton_sw_deflection_angle)),
            DataProductVariable(PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.proton_sw_deflection_angle))
        ]


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    alpha_sw_speed: np.ndarray[float]
    alpha_sw_temperature: np.ndarray[float]
    alpha_sw_density: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, THIRTY_SECONDS_IN_NANOSECONDS, record_varying=False),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.alpha_sw_speed)),
            DataProductVariable(ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.alpha_sw_speed)),
            DataProductVariable(ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.alpha_sw_temperature)),
            DataProductVariable(ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME,
                                std_devs(self.alpha_sw_temperature)),
            DataProductVariable(ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME, nominal_values(self.alpha_sw_density)),
            DataProductVariable(ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.alpha_sw_density))
        ]


@dataclass
class SwapiL3PickupIonData(DataProduct):
    epoch: np.ndarray[float]
    cooling_index: np.ndarray[float]
    ionization_rate: np.ndarray[float]
    cutoff_speed: np.ndarray[float]
    background_rate: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, FIVE_MINUTES_IN_NANOSECONDS, record_varying=False),
            DataProductVariable(PUI_COOLING_INDEX_CDF_VAR_NAME, self.cooling_index),
            DataProductVariable(PUI_IONIZATION_RATE_CDF_VAR_NAME, self.ionization_rate),
            DataProductVariable(PUI_CUTOFF_SPEED_CDF_VAR_NAME, self.cutoff_speed),
            DataProductVariable(PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME, self.background_rate),
        ]


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    spin_angles: np.ndarray[float]  # not currently in the L2 cdf, is in the sample data provided by Bishwas
    coincidence_count_rate_uncertainty: np.ndarray[float]
