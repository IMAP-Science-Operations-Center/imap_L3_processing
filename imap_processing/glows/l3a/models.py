from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class GlowsL2Data:
    epoch: datetime
    start_time: datetime
    end_time: datetime
    histogram_flags_array: np.ndarray[int]
    photon_flux: np.ndarray[float]
    flux_uncertainties: np.ndarray[float]
    spin_angles: np.ndarray[float]
    exposure_times: np.ndarray[float]
