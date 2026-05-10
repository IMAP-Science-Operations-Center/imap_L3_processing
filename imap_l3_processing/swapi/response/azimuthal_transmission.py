import math
from typing import NamedTuple

import numba
from numpy import ndarray


class AzimuthalTransmissionGrid(NamedTuple):
    """Azimuth-only transmission curve `T(|az|)` sampled at uniform spacing.

    The curve is symmetric about az=0 (the table stores only non-negative
    indices; callers feed signed azimuths and the interpolator mirrors)."""

    values: ndarray
    spacing: float


@numba.njit
def interpolate_azimuthal_transmission(
    grid: AzimuthalTransmissionGrid,
    azimuth: float,
) -> float:
    """Linear interpolation of the (azimuth-only) transmission curve, with
    wrap-around to ±180° and a symmetric extension across azimuth=0."""
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
