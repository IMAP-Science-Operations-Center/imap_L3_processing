import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable, UpstreamDataDependency

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

    @classmethod
    def convert_dat_to_glows_l3e_hi_product(cls, input_metadata: UpstreamDataDependency, file_path: Path,
                                            epoch: np.ndarray[datetime],
                                            epoch_delta: np.ndarray[timedelta]):
        with open(file_path) as input_data:
            energy_line = [line for line in input_data.readlines() if line.startswith("#energy_grid")]
            energies = np.array([float(i) for i in re.findall(r"\d+.\d+", energy_line[0])])

        spin_angle_and_survival_probabilities = np.loadtxt(file_path, skiprows=200)
        spin_angles = spin_angle_and_survival_probabilities[:, 0]
        survival_probabilities = np.array([spin_angle_and_survival_probabilities[:, 1:].T])

        return cls(input_metadata, epoch, epoch_delta, energies, spin_angles, survival_probabilities)

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(SPIN_ANGLE_VAR_NAME, self.spin_angle),
            DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, self.probability_of_survival)
        ]
