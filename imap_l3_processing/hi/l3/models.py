from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class HiL1cData:
    epoch: datetime
    epoch_j2000: np.ndarray
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray


@dataclass
class HiGlowsL3eData:
    epoch: datetime
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray
