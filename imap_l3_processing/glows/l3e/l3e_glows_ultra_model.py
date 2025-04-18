from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "probability_of_survival"
LATITUDE_VAR_NAME = "latitude"
LONGITUDE_VAR_NAME = "longitude"
HEALPIX_INDEX_VAR_NAME = "healpix_index"


@dataclass
class GlowsL3EUltraData(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[timedelta]
    energy: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    healpix_index: np.ndarray
    probability_of_survival: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(LATITUDE_VAR_NAME, self.latitude),
            DataProductVariable(LONGITUDE_VAR_NAME, self.longitude),
            DataProductVariable(HEALPIX_INDEX_VAR_NAME, self.healpix_index),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival)
        ]
