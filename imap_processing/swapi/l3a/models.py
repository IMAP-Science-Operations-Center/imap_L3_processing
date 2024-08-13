from dataclasses import dataclass

import numpy as np


@dataclass
class SwapiL3Data:
    epoch: np.ndarray[float]
    proton_sw_speed: np.ndarray[float]


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    spin_angles: np.ndarray[float]  # not currently in the L2 cdf, is in the sample data provided by Bishwas