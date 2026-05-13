import math
from typing import NamedTuple

import numba
from numpy import ndarray


class AzimuthalTransmissionGrid(NamedTuple):
    values: ndarray
    spacing: float


@numba.njit
def interpolate_azimuthal_transmission(
    grid: AzimuthalTransmissionGrid,
    azimuth: float,
) -> float:
    azimuth = (azimuth + 180) % 360 - 180
    i_float = abs(azimuth) / grid.spacing
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
