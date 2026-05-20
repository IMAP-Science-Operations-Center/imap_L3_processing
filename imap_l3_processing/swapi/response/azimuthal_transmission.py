import math
from typing import NamedTuple

import numba
import numpy as np
from numpy import ndarray


SG_PLATEAU_AZIMUTH_MAX_DEG = 9.0
SG_PLATEAU_TRANSMISSION = 1 / 1000

OA_PLATEAU_AZIMUTH_MIN_DEG = 31.0
OA_PLATEAU_AZIMUTH_MAX_DEG = 115.0
OA_PLATEAU_TRANSMISSION = 1.0


class AzimuthalTransmissionGrid(NamedTuple):
    values: ndarray
    spacing: float


@numba.njit
def interpolate_azimuthal_transmission(
    grid: AzimuthalTransmissionGrid,
    azimuth: float,
) -> float:
    azimuth = (azimuth + 180) % 360 - 180
    abs_azimuth = abs(azimuth)

    if OA_PLATEAU_AZIMUTH_MIN_DEG <= abs_azimuth <= OA_PLATEAU_AZIMUTH_MAX_DEG:
        return OA_PLATEAU_TRANSMISSION
    if abs_azimuth <= SG_PLATEAU_AZIMUTH_MAX_DEG:
        return SG_PLATEAU_TRANSMISSION

    i_float = abs_azimuth / grid.spacing
    i_lower = int(math.floor(i_float))
    i_upper = i_lower + 1

    n = len(grid.values)
    if i_lower >= n:
        i_lower = n - 1
    if i_upper >= n:
        i_upper = n - 1

    weight_lower = float(i_upper) - i_float
    weight_upper = i_float - float(i_lower)
    return (
        grid.values[i_lower] * weight_lower
        + grid.values[i_upper] * weight_upper
    )


def validate_azimuthal_transmission_values(values: ndarray, spacing: float) -> None:
    values = np.asarray(values, dtype=float)
    azimuths = np.arange(values.size) * spacing

    in_sg_plateau = azimuths <= SG_PLATEAU_AZIMUTH_MAX_DEG
    if not np.allclose(values[in_sg_plateau], SG_PLATEAU_TRANSMISSION):
        raise ValueError(
            f"Azimuthal transmission values for |az| <= {SG_PLATEAU_AZIMUTH_MAX_DEG}° "
            f"must all equal {SG_PLATEAU_TRANSMISSION} "
            f"(SG plateau assumed by interpolate_azimuthal_transmission)."
        )

    in_oa_plateau = (azimuths >= OA_PLATEAU_AZIMUTH_MIN_DEG) & (
        azimuths <= OA_PLATEAU_AZIMUTH_MAX_DEG
    )
    if not np.allclose(values[in_oa_plateau], OA_PLATEAU_TRANSMISSION):
        raise ValueError(
            f"Azimuthal transmission values for {OA_PLATEAU_AZIMUTH_MIN_DEG}° <= |az| "
            f"<= {OA_PLATEAU_AZIMUTH_MAX_DEG}° must all equal {OA_PLATEAU_TRANSMISSION} "
            f"(OA plateau assumed by interpolate_azimuthal_transmission)."
        )
