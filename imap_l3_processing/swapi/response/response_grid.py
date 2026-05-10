import math
from typing import NamedTuple

import numba
from numpy import ndarray

from imap_l3_processing.swapi.response.passband_grid import PassbandGrid


class ResponseGrid(NamedTuple):
    """V-and-species-specific instrument response evaluated at one ESA step.

    `azimuthal_transmission` and `azimuthal_transmission_spacing` are the same
    array reference across all grids in a sweep — bundling them per-grid is
    pointer-cheap and keeps the integration call signature compact."""

    passband_grid: PassbandGrid
    central_speed: float
    central_effective_area: float
    azimuthal_transmission: ndarray
    azimuthal_transmission_spacing: float


@numba.njit
def interpolate_azimuthal_transmission(
    azimuthal_transmission,
    azimuthal_transmission_spacing: float,
    azimuth: float,
) -> float:
    """Linear interpolation of the (azimuth-only) transmission curve, with
    wrap-around to ±180° and a symmetric extension across azimuth=0."""
    azimuth = (azimuth + 180) % 360 - 180
    i_float = abs(azimuth) / azimuthal_transmission_spacing
    i_lower = int(math.floor(i_float))
    i_upper = i_lower + 1

    n = len(azimuthal_transmission)
    if i_lower >= n:
        i_lower = n - 1
    if i_upper >= n:
        i_upper = n - 1

    weight_lower = float(i_upper) - i_float
    weight_upper = i_float - float(i_lower)
    return (
        azimuthal_transmission[i_lower] * weight_lower
        + azimuthal_transmission[i_upper] * weight_upper
    )
