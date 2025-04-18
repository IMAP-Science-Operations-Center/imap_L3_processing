from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
SPIN_ANGLE_VAR_NAME = "spin_angle"
PROBABILITY_OF_SURVIVAL_VAR_NAME = "probability_of_survival"


@dataclass
class GlowsL3EHiData(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[timedelta]
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival)
        ]
