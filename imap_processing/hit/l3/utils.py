import math

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf: CDF) -> HitL2Data:
    return HitL2Data(
        epoch=cdf.raw_var("Epoch")[...],
        epoch_delta=cdf["Epoch_DELTA"][...],
        flux=cdf["R26A_H_SECT_Flux"][...],
        count_rates=cdf["R26A_H_SECT_Rate"][...],
        uncertainty=cdf["R26A_H_SECT_Uncertainty"][...],
    )


def calculate_pitch_angle(x: np.ndarray[float], y: np.ndarray[float]) -> float:
    if len(x) != len(y):
        raise ValueError(f"Input vectors are of unequal length {len(x)} and {len(y)}")
    return np.degrees(math.acos(np.dot(x, y)))


def calculate_unit_vector(vector: np.ndarray[float]) -> np.ndarray[float]:
    return vector / np.linalg.norm(vector)
