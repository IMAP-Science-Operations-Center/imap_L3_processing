from typing import NamedTuple

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from imap_l3_processing.swapi.response.speed_calculation import SWAPI_K_FACTOR

_TARGET_ELEVATIONS = np.arange(-15, 15 + 0.5, 0.5)
_TARGET_SPEED_RATIOS = np.linspace(0.9, 1.1, 101)
_PASSBAND_BOUNDARY_THRESHOLD = 1e-2


class PassbandGrid(NamedTuple):
    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values: NDArray
    min_boundary: NDArray
    max_boundary: NDArray
    elevation_range: tuple


def build_passband_grid(values_df: pd.DataFrame) -> PassbandGrid:
    grid_values = _build_passband_array(
        values_df, _TARGET_ELEVATIONS, _TARGET_SPEED_RATIOS
    )
    min_boundary, max_boundary = _passband_boundaries(grid_values, _TARGET_SPEED_RATIOS)
    elevation_range = _elevation_range(grid_values, _TARGET_ELEVATIONS)

    return PassbandGrid(
        min_elevation=float(_TARGET_ELEVATIONS[0]),
        elevation_spacing=float(_TARGET_ELEVATIONS[1] - _TARGET_ELEVATIONS[0]),
        min_speed_ratio=float(_TARGET_SPEED_RATIOS[0]),
        speed_ratio_spacing=float(_TARGET_SPEED_RATIOS[1] - _TARGET_SPEED_RATIOS[0]),
        values=grid_values,
        min_boundary=min_boundary,
        max_boundary=max_boundary,
        elevation_range=elevation_range,
    )


@numba.njit
def speed_ratio_range_at_elevation(grid, elevation: float):
    n_elevations = grid.min_boundary.shape[0]
    i_float = (elevation - grid.min_elevation) / grid.elevation_spacing
    if i_float < 0.0:
        lower_row = 0
    elif i_float >= n_elevations - 1:
        lower_row = n_elevations - 1
    else:
        lower_row = int(i_float)
    upper_row = lower_row + 1 if lower_row + 1 < n_elevations else n_elevations - 1

    lower_min = grid.min_boundary[lower_row]
    upper_min = grid.min_boundary[upper_row]
    lower_max = grid.max_boundary[lower_row]
    upper_max = grid.max_boundary[upper_row]
    # If only one bracketing row is below the active-range cutoff (NaN), drop
    # it — the union is just the valid row's interval. If both are NaN the
    # caller queried outside the active range and failed to gate on
    # `elevation_range`.
    lower_is_nan = np.isnan(lower_min) or np.isnan(lower_max)
    upper_is_nan = np.isnan(upper_min) or np.isnan(upper_max)
    if lower_is_nan and upper_is_nan:
        raise ValueError("Elevation out of range.")
    if lower_is_nan:
        return upper_min, upper_max
    if upper_is_nan:
        return lower_min, lower_max
    lo = lower_min if lower_min < upper_min else upper_min
    hi = lower_max if lower_max > upper_max else upper_max
    return lo, hi


@numba.njit
def interpolate_passband(grid, elevation: float, speed_ratio: float) -> float:
    return _bilinear_interpolate(
        values=grid.values,
        x0=grid.min_speed_ratio,
        dx=grid.speed_ratio_spacing,
        y0=grid.min_elevation,
        dy=grid.elevation_spacing,
        x=speed_ratio,
        y=elevation,
    )


@numba.njit
def _bilinear_interpolate(
    values: NDArray,
    x0: float,
    dx: float,
    y0: float,
    dy: float,
    x: float,
    y: float,
) -> float:
    i_float = (y - y0) / dy
    if i_float < 0 or i_float + 1 >= values.shape[0]:
        return 0.0
    j_float = (x - x0) / dx
    if j_float < 0 or j_float + 1 >= values.shape[1]:
        return 0.0

    i_lower = int(i_float)
    i_upper = i_lower + 1
    i_weight = i_float - i_lower
    j_lower = int(j_float)
    j_upper = j_lower + 1
    j_weight = j_float - j_lower

    return (1 - i_weight) * (
        (1 - j_weight) * values[i_lower, j_lower] + j_weight * values[i_lower, j_upper]
    ) + i_weight * (
        (1 - j_weight) * values[i_upper, j_lower] + j_weight * values[i_upper, j_upper]
    )


def _build_passband_array(
    values_df: pd.DataFrame, target_elevations: NDArray, target_speed_ratios: NDArray
) -> NDArray:
    pivot = values_df.reset_index()
    pivot["speed_ratio"] = np.sqrt(pivot["energy_ratio"] / SWAPI_K_FACTOR)
    pivot = (
        pivot.drop(columns="energy_ratio")
        .set_index(["elevation", "speed_ratio"])["value"]
        .unstack("speed_ratio")
    )
    pivot = pivot.fillna(0.0)

    src_speed_ratios = pivot.columns.values
    result = np.zeros((len(target_elevations), len(target_speed_ratios)))
    for i, elev in enumerate(target_elevations):
        if elev in pivot.index:
            result[i] = np.interp(
                target_speed_ratios,
                src_speed_ratios,
                pivot.loc[elev].values,
                left=0.0,
                right=0.0,
            )

    # Normalize so the central value (elevation=0, speed_ratio=1) is 1.
    central_value = _bilinear_interpolate(
        values=result,
        x0=target_speed_ratios[0],
        dx=target_speed_ratios[1] - target_speed_ratios[0],
        y0=target_elevations[0],
        dy=target_elevations[1] - target_elevations[0],
        x=1.0,
        y=0.0,
    )
    result = result / central_value
    return result.copy(order="C")


def _passband_boundaries(
    grid_values: NDArray, target_speed_ratios: NDArray
) -> tuple[NDArray, NDArray]:
    cutoff = _cutoff_value(grid_values)
    n_elevations = grid_values.shape[0]
    mins = np.full(n_elevations, np.nan)
    maxs = np.full(n_elevations, np.nan)
    for i in range(n_elevations):
        above = np.flatnonzero(grid_values[i] >= cutoff)
        if above.size == 0:
            continue
        mins[i] = _interp_cutoff_crossing(
            target_speed_ratios, grid_values[i], cutoff, above[0], above[0] - 1
        )
        maxs[i] = _interp_cutoff_crossing(
            target_speed_ratios, grid_values[i], cutoff, above[-1], above[-1] + 1
        )
    return mins, maxs


def _elevation_range(grid_values: NDArray, target_elevations: NDArray) -> tuple:
    cutoff = _cutoff_value(grid_values)
    row_max = grid_values.max(axis=1)
    above = np.flatnonzero(row_max >= cutoff)

    lo = _interp_cutoff_crossing(
        target_elevations, row_max, cutoff, above[0], above[0] - 1
    )
    hi = _interp_cutoff_crossing(
        target_elevations, row_max, cutoff, above[-1], above[-1] + 1
    )
    return (float(lo), float(hi))


def _interp_cutoff_crossing(
    x: NDArray, y: NDArray, cutoff: float, i_inside: int, i_outside: int
) -> float:
    """Linear interpolation of `x` where `y` crosses `cutoff`, given that
    `y[i_inside] >= cutoff` and `y[i_outside] < cutoff`."""
    return x[i_inside] + (cutoff - y[i_inside]) / (y[i_outside] - y[i_inside]) * (
        x[i_outside] - x[i_inside]
    )


def _cutoff_value(grid_values):
    return _PASSBAND_BOUNDARY_THRESHOLD * float(grid_values.max())
