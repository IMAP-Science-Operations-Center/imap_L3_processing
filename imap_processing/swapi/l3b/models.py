from dataclasses import dataclass

import numpy as np
from spacepy import pycdf
from uncertainties.unumpy import nominal_values, std_devs

from imap_processing.constants import FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.models import DataProduct, DataProductVariable
from imap_processing.swapi.l3a.models import EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME

PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME = "proton_sw_velocity"
PROTON_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME = "proton_sw_velocity_delta"

PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME = "proton_sw_vdf"
PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME = "proton_sw_vdf_delta"

ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME = "alpha_sw_velocity"
ALPHA_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME = "alpha_sw_velocity_delta"

ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME = "alpha_sw_vdf"
ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME = "alpha_sw_vdf_delta"


@dataclass
class SwapiL3BCombinedVDF(DataProduct):
    epoch: np.ndarray[float]
    proton_sw_velocities: np.ndarray[float]
    proton_sw_combined_vdf: np.ndarray[float]
    alpha_sw_velocities: np.ndarray[float]
    alpha_sw_combined_vdf: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, FIVE_MINUTES_IN_NANOSECONDS, record_varying=False),

            DataProductVariable(PROTON_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, nominal_values(self.proton_sw_velocities)),
            DataProductVariable(PROTON_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME, 6.0, record_varying=False),

            DataProductVariable(PROTON_SOLAR_WIND_VDF_CDF_VAR_NAME, nominal_values(self.proton_sw_combined_vdf)),
            DataProductVariable(PROTON_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, std_devs(self.proton_sw_combined_vdf)),

            DataProductVariable(ALPHA_SOLAR_WIND_VELOCITIES_CDF_VAR_NAME, nominal_values(self.alpha_sw_velocities)),
            DataProductVariable(ALPHA_SOLAR_WIND_VELOCITIES_DELTAS_CDF_VAR_NAME, 6.0, record_varying=False),

            DataProductVariable(ALPHA_SOLAR_WIND_VDF_CDF_VAR_NAME, nominal_values(self.alpha_sw_combined_vdf)),
            DataProductVariable(ALPHA_SOLAR_WIND_VDF_DELTAS_CDF_VAR_NAME, std_devs(self.alpha_sw_combined_vdf)),
        ]
