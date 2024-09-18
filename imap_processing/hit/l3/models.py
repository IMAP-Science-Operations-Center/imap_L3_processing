from dataclasses import dataclass

import numpy as np


@dataclass
class HitL2Data:
    epoch: np.ndarray[float]
    epoch_delta: np.ndarray[float]
    flux: np.ndarray[float]
    count_rates: np.ndarray[float]
    uncertainty: np.ndarray[float]


