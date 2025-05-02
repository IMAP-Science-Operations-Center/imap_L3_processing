from dataclasses import dataclass

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable

LAT_GRID_CDF_VAR_NAME = "latitude_grid"
CR_GRID_CDF_VAR_NAME = "cr_grid"
TIME_GRID_CDF_VAR_NAME = "time_grid"
SPEED_CDF_VAR_NAME = "speed"
P_DENS_CDF_VAR_NAME = "proton_density"
UV_ANIS_CDF_VAR_NAME = "uv_anisotropy"
PHION_CDF_VAR_NAME = "phion"
LYA_CDF_VAR_NAME = "lyman_alpha"
E_DENS_CDF_VAR_NAME = "electron_density"


@dataclass
class GlowsL3DSolarParamsHistory(DataProduct):
    lat_grid: np.ndarray
    cr_grid: np.ndarray
    time_grid: np.ndarray
    speed: np.ndarray
    p_dens: np.ndarray
    uv_anis: np.ndarray
    phion: np.ndarray
    lya: np.ndarray
    e_dens: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(LAT_GRID_CDF_VAR_NAME, self.lat_grid),
            DataProductVariable(CR_GRID_CDF_VAR_NAME, self.cr_grid),
            DataProductVariable(TIME_GRID_CDF_VAR_NAME, self.time_grid),
            DataProductVariable(SPEED_CDF_VAR_NAME, self.speed),
            DataProductVariable(P_DENS_CDF_VAR_NAME, self.p_dens),
            DataProductVariable(UV_ANIS_CDF_VAR_NAME, self.uv_anis),
            DataProductVariable(PHION_CDF_VAR_NAME, self.phion),
            DataProductVariable(LYA_CDF_VAR_NAME, self.lya),
            DataProductVariable(E_DENS_CDF_VAR_NAME, self.e_dens),
        ]
