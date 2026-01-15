from dataclasses import dataclass
from datetime import datetime

import numpy as np
from uncertainties.unumpy import nominal_values, std_devs

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME = "proton_sw_speed"
PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_speed_uncert"
PROTON_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "proton_sw_temperature"
PROTON_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_temperature_uncert"
PROTON_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "proton_sw_density"
PROTON_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_density_uncert"

PROTON_SOLAR_WIND_CLOCK_ANGLE_CDF_VAR_NAME = "proton_sw_clock_angle"
PROTON_SOLAR_WIND_CLOCK_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_clock_angle_uncert"

PROTON_SOLAR_WIND_DEFLECTION_ANGLE_CDF_VAR_NAME = "proton_sw_deflection_angle"
PROTON_SOLAR_WIND_DEFLECTION_ANGLE_UNCERTAINTY_CDF_VAR_NAME = "proton_sw_deflection_angle_uncert"

ALPHA_SOLAR_WIND_SPEED_CDF_VAR_NAME = "alpha_sw_speed"
ALPHA_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_speed_uncert"
ALPHA_SOLAR_WIND_DENSITY_CDF_VAR_NAME = "alpha_sw_density"
ALPHA_SOLAR_WIND_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_density_uncert"
ALPHA_SOLAR_WIND_TEMPERATURE_CDF_VAR_NAME = "alpha_sw_temperature"
ALPHA_SOLAR_WIND_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "alpha_sw_temperature_uncert"

PUI_COOLING_INDEX_CDF_VAR_NAME = "pui_cooling_index"
PUI_IONIZATION_RATE_CDF_VAR_NAME = "pui_ionization_rate"
PUI_CUTOFF_SPEED_CDF_VAR_NAME = "pui_cutoff_speed"
PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME = "pui_background_count_rate"
PUI_DENSITY_CDF_VAR_NAME = "pui_density"
PUI_TEMPERATURE_CDF_VAR_NAME = "pui_temperature"
PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME = "pui_cooling_index_uncert"
PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME = "pui_ionization_rate_uncert"
PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME = "pui_cutoff_speed_uncert"
PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME = "pui_background_count_rate_uncert"
PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME = "pui_density_uncert"
PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME = "pui_temperature_uncert"

SWAPI_QUALITY_FLAGS_CDF_VAR_NAME = "swp_flags"

@dataclass
class SwapiL3ProtonSolarWindData(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]
    proton_sw_temperature: np.ndarray[float]
    proton_sw_density: np.ndarray[float]
    proton_sw_clock_angle: np.ndarray[float]
    proton_sw_deflection_angle: np.ndarray[float]
    quality_flags: np.ndarray[SwapiL3Flags]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_CDF_VAR_NAME, nominal_values(self.proton_sw_speed)),
            DataProductVariable(PROTON_SOLAR_WIND_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.proton_sw_speed)),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS)),
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
                                std_devs(self.proton_sw_deflection_angle)),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags)
        ]


@dataclass
class SwapiL3AlphaSolarWindData(DataProduct):
    epoch: np.ndarray[datetime]
    alpha_sw_speed: np.ndarray[float]
    alpha_sw_temperature: np.ndarray[float]
    alpha_sw_density: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, THIRTY_SECONDS_IN_NANOSECONDS)),
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
    density: np.ndarray[float]
    temperature: np.ndarray[float]
    quality_flags: np.ndarray[SwapiL3Flags]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, np.full_like(self.epoch, FIVE_MINUTES_IN_NANOSECONDS)),
            DataProductVariable(PUI_COOLING_INDEX_CDF_VAR_NAME, nominal_values(self.cooling_index)),
            DataProductVariable(PUI_COOLING_INDEX_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cooling_index)),
            DataProductVariable(PUI_IONIZATION_RATE_CDF_VAR_NAME, nominal_values(self.ionization_rate)),
            DataProductVariable(PUI_IONIZATION_RATE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.ionization_rate)),
            DataProductVariable(PUI_CUTOFF_SPEED_CDF_VAR_NAME, nominal_values(self.cutoff_speed)),
            DataProductVariable(PUI_CUTOFF_SPEED_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.cutoff_speed)),
            DataProductVariable(PUI_BACKGROUND_COUNT_RATE_CDF_VAR_NAME, nominal_values(self.background_rate)),
            DataProductVariable(PUI_BACKGROUND_COUNT_RATE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.background_rate)),
            DataProductVariable(PUI_DENSITY_CDF_VAR_NAME, nominal_values(self.density)),
            DataProductVariable(PUI_DENSITY_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.density)),
            DataProductVariable(PUI_TEMPERATURE_CDF_VAR_NAME, nominal_values(self.temperature)),
            DataProductVariable(PUI_TEMPERATURE_UNCERTAINTY_CDF_VAR_NAME, std_devs(self.temperature)),
            DataProductVariable(SWAPI_QUALITY_FLAGS_CDF_VAR_NAME, self.quality_flags),
        ]


@dataclass
class SwapiL2Data:
    sci_start_time: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    coincidence_count_rate_uncertainty: np.ndarray[float]
