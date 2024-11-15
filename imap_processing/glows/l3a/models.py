from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class GlowsL2Data:
    start_time: datetime
    end_time: datetime
    histogram_flag_array: np.ndarray[bool]
    photon_flux: np.ndarray[float]
    flux_uncertainties: np.ndarray[float]
    spin_angle: np.ndarray[float]
    exposure_times: np.ndarray[float]
