from dataclasses import dataclass

import numpy as np

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = 'epoch'
EPOCH_DELTA_CDF_VAR_NAME = 'epoch_delta'
LATITUDE_CDF_CDF_VAR_NAME = "latitude"
LATITUDE_LABEL_CDF_VAR_NAME = "latitude_label"
CR_CDF_VAR_NAME = "cr"
SPEED_CDF_VAR_NAME = "plasma_speed"
PROTON_DENSITY_CDF_VAR_NAME = "proton_density"
UV_ANISOTROPY_CDF_VAR_NAME = "ultraviolet_anisotropy"
PHION_CDF_VAR_NAME = "phion"
LYMAN_ALPHA_CDF_VAR_NAME = "lyman_alpha"
ELECTRON_DENSITY_CDF_VAR_NAME = "electron_density"
PLASMA_SPEED_FLAG_CDF_VAR_NAME = "plasma_speed_flag"
UV_ANISOTROPY_FLAG_CDF_VAR_NAME = "uv_anisotropy_flag"
PROTON_DENSITY_FLAG_CDF_VAR_NAME = "proton_density_flag"


@dataclass
class GlowsL3DSolarParamsHistory(DataProduct):
    epoch: np.ndarray
    latitude: np.ndarray
    cr: np.ndarray
    plasma_speed: np.ndarray
    proton_density: np.ndarray
    ultraviolet_anisotropy: np.ndarray
    phion: np.ndarray
    lyman_alpha: np.ndarray
    electron_density: np.ndarray
    plasma_speed_flag: np.ndarray
    uv_anisotropy_flag: np.ndarray
    proton_density_flag: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        epoch_delta = np.full_like(self.epoch, CARRINGTON_ROTATION_IN_NANOSECONDS / 2)
        latitude_label = [f'{lat:.1f} degrees' for lat in self.latitude]

        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, epoch_delta),
            DataProductVariable(LATITUDE_CDF_CDF_VAR_NAME, self.latitude),
            DataProductVariable(LATITUDE_LABEL_CDF_VAR_NAME, latitude_label),
            DataProductVariable(CR_CDF_VAR_NAME, self.cr),
            DataProductVariable(SPEED_CDF_VAR_NAME, self.plasma_speed),
            DataProductVariable(PROTON_DENSITY_CDF_VAR_NAME, self.proton_density),
            DataProductVariable(UV_ANISOTROPY_CDF_VAR_NAME, self.ultraviolet_anisotropy),
            DataProductVariable(PHION_CDF_VAR_NAME, self.phion),
            DataProductVariable(LYMAN_ALPHA_CDF_VAR_NAME, self.lyman_alpha),
            DataProductVariable(ELECTRON_DENSITY_CDF_VAR_NAME, self.electron_density),
            DataProductVariable(PLASMA_SPEED_FLAG_CDF_VAR_NAME, self.plasma_speed_flag),
            DataProductVariable(UV_ANISOTROPY_FLAG_CDF_VAR_NAME, self.uv_anisotropy_flag),
            DataProductVariable(PROTON_DENSITY_FLAG_CDF_VAR_NAME, self.proton_density_flag),
        ]
