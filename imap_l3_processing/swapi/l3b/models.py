from dataclasses import dataclass

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import nominal_values, std_devs

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable
from imap_l3_processing.swapi.l3a.models import EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME

PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME = "proton_sw_velocity"
PROTON_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME = "proton_sw_velocity_delta_minus"
PROTON_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME = "proton_sw_velocity_delta_plus"

PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME = "proton_sw_vdf"
PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME = "proton_sw_vdf_delta"

ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME = "alpha_sw_velocity"
ALPHA_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME = "alpha_sw_velocity_delta_minus"
ALPHA_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME = "alpha_sw_velocity_delta_plus"

ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME = "alpha_sw_vdf"
ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME = "alpha_sw_vdf_delta"

PUI_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME = "pui_sw_velocity"
PUI_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME = "pui_sw_velocity_delta_minus"
PUI_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME = "pui_sw_velocity_delta_plus"

PUI_SOLAR_WIND_VDF_CDF_VAR_NAME = "pui_sw_vdf"
PUI_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME = "pui_sw_vdf_delta"

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
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, FIVE_MINUTES_IN_NANOSECONDS, record_varying=False),

            DataProductVariable(PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, nominal_values(self.proton_sw_velocities)),
            DataProductVariable(PROTON_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME,
                                self.proton_sw_velocities_delta_minus),
            DataProductVariable(PROTON_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME,
                                self.proton_sw_velocities_delta_plus),

            DataProductVariable(PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME, nominal_values(self.proton_sw_combined_vdf)),
            DataProductVariable(PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, std_devs(self.proton_sw_combined_vdf)),

            DataProductVariable(ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, nominal_values(self.alpha_sw_velocities)),
            DataProductVariable(ALPHA_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME,
                                self.alpha_sw_velocities_delta_minus),
            DataProductVariable(ALPHA_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME,
                                self.alpha_sw_velocities_delta_plus),

            DataProductVariable(ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME, nominal_values(self.alpha_sw_combined_vdf)),
            DataProductVariable(ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, std_devs(self.alpha_sw_combined_vdf)),

            DataProductVariable(PUI_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, nominal_values(self.pui_sw_velocities)),
            DataProductVariable(PUI_SOLAR_WIND_VELOCITIES_DELTA_MINUS_CDF_VAR_NAME, self.pui_sw_velocities_delta_minus),
            DataProductVariable(PUI_SOLAR_WIND_VELOCITIES_DELTA_PLUS_CDF_VAR_NAME, self.pui_sw_velocities_delta_plus),

            DataProductVariable(PUI_SOLAR_WIND_VDF_CDF_VAR_NAME, nominal_values(self.pui_sw_combined_vdf)),
            DataProductVariable(PUI_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, std_devs(self.pui_sw_combined_vdf)),

            DataProductVariable(SOLAR_WIND_ENERGY_CDF_VAR_NAME, self.combined_energy),
            DataProductVariable(SOLAR_WIND_COMBINED_ENERGY_DELTA_MINUS_CDF_VAR_NAME, self.combined_energy_delta_minus),
            DataProductVariable(SOLAR_WIND_COMBINED_ENERGY_DELTA_PLUS_CDF_VAR_NAME, self.combined_energy_delta_plus),

            DataProductVariable(COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_CDF_VAR_NAME,
                                nominal_values(self.combined_differential_flux)),
            DataProductVariable(COMBINED_SOLAR_WIND_DIFFERENTIAL_FLUX_DELTA_CDF_VAR_NAME,
                                std_devs(self.combined_differential_flux))
        ]
