import math

import numba
import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    SolarWindParams,
    thermal_speed,
)
from imap_l3_processing.swapi.response.response_grid import ResponseGrid


def average_spin_axis_rtn(rotation_matrices: ndarray) -> ndarray:
    boresight_direction_in_RTN = rotation_matrices[:, :, 1]
    mean_direction = boresight_direction_in_RTN.mean(axis=0)
    normalized_mean_direction = mean_direction / np.linalg.norm(mean_direction)
    return normalized_mean_direction


@numba.njit
def count_rate_conversion_factor(
    sw_params: SolarWindParams, response_grid: ResponseGrid
) -> float:
    sigma = thermal_speed(sw_params)
    return (
        response_grid.central_effective_area
        * sw_params.density
        * (np.sqrt(2 * np.pi) * sigma) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # km -> cm
    )
