from typing import NamedTuple

import numba
import numpy as np

from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    Region,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.open_aperture_trimming import (
    trim_oa_azimuth_by_integrand,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
    bulk_angles_in_instrument_frame,
    bulk_speed,
    thermal_speed,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_l3_processing.swapi.response.passband_grid import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid


class AngularQuadrature(NamedTuple):
    elevation_points: np.ndarray
    elevation_weights: np.ndarray
    azimuth_points: np.ndarray
    azimuth_weights: np.ndarray
    sin_elevation: np.ndarray
    cos_elevation: np.ndarray
    sin_azimuth: np.ndarray
    cos_azimuth: np.ndarray
    transmission_azimuth: np.ndarray


class SpeedQuadrature(NamedTuple):
    points: np.ndarray
    weights: np.ndarray
    # per-node `v³ × passband(v/central_speed)`, normalized by the on-axis peak
    speed_cubed_times_passband: np.ndarray


_GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION = np.polynomial.legendre.leggauss(21)
_GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH = np.polynomial.legendre.leggauss(21)
_GL_NODES_SPEED, _GL_WEIGHTS_SPEED = np.polynomial.legendre.leggauss(15)


# Threshold on the bin-relative Maxwellian falloff used to size the angular
# integration window around the bulk direction. See `solar-wind-moments.md`.
EPSILON = 1e-6

# Half-width (in thermal speeds) of the speed-axis integration window. See
# `solar-wind-moments.md` for the empirical sweep that selected k=6.
SPEED_HALF_WIDTH_VTH = 6.0


@numba.njit
def speed_window_misses_passband(
    sw_params: SolarWindParams, response_grid: ResponseGrid
) -> bool:
    sigma = thermal_speed(sw_params)
    speed = bulk_speed(sw_params)

    sw_window_lo = speed - SPEED_HALF_WIDTH_VTH * sigma
    sw_window_hi = speed + SPEED_HALF_WIDTH_VTH * sigma

    central_speed = response_grid.central_speed
    # union of SG and OA passband speed ranges, evaluated on-axis (elevation=0)
    sg_lo, sg_hi = speed_ratio_range_at_elevation(response_grid.sg_passband, 0.0)
    oa_lo, oa_hi = speed_ratio_range_at_elevation(response_grid.oa_passband, 0.0)
    passband_lo = central_speed * min(sg_lo, oa_lo)
    passband_hi = central_speed * max(sg_hi, oa_hi)
    return sw_window_hi < passband_lo or sw_window_lo > passband_hi


@numba.njit
def get_angular_quadrature(
    sw_params: SolarWindParams,
    response_grid: ResponseGrid,
    region: Region,
    rotation_matrix,
    sg_rate: float,
):
    # `sg_rate` is the SG rate already accumulated this ESA step (used only for
    # the OA skip threshold below); 0.0 is fine for SG/VV calls.
    min_el, max_el, min_az, max_az = _angular_limits(
        sw_params, rotation_matrix, region, response_grid
    )
    if max_el <= min_el or max_az <= min_az:
        return True, None

    if region.is_open_aperture:
        min_az, max_az = trim_oa_azimuth_by_integrand(
            response_grid,
            sw_params,
            rotation_matrix,
            min_el,
            max_el,
            min_az,
            max_az,
            sg_rate,
        )
        if max_az <= min_az:
            return True, None

    def rescale(nodes, weights, lower, upper):
        half = 0.5 * (upper - lower)
        mid = 0.5 * (upper + lower)
        return mid + half * nodes, half * weights

    elevation_points, elevation_weights = rescale(
        _GL_NODES_ELEVATION, _GL_WEIGHTS_ELEVATION, min_el, max_el
    )
    azimuth_points, azimuth_weights = rescale(
        _GL_NODES_AZIMUTH, _GL_WEIGHTS_AZIMUTH, min_az, max_az
    )
    transmission_azimuth = np.array(
        [
            interpolate_azimuthal_transmission(
                response_grid.azimuthal_transmission, az
            )
            for az in azimuth_points
        ]
    )
    return False, AngularQuadrature(
        elevation_points=elevation_points,
        elevation_weights=elevation_weights,
        azimuth_points=azimuth_points,
        azimuth_weights=azimuth_weights,
        sin_elevation=np.sin(np.radians(elevation_points)),
        cos_elevation=np.cos(np.radians(elevation_points)),
        sin_azimuth=np.sin(np.radians(azimuth_points)),
        cos_azimuth=np.cos(np.radians(azimuth_points)),
        transmission_azimuth=transmission_azimuth,
    )


@numba.njit
def get_speed_quadrature(
    sw_params: SolarWindParams,
    response_grid: ResponseGrid,
    region: Region,
    elevation: float,
):
    passband = (
        response_grid.sg_passband if region.is_sunglasses else response_grid.oa_passband
    )
    central_speed = response_grid.central_speed
    sigma = thermal_speed(sw_params)
    bulk_v = bulk_speed(sw_params)

    # `bulk_v ± k·σ` Maxwellian window intersected with the region's passband at this elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(passband, elevation)
    passband_lo = central_speed * ratio_lo
    passband_hi = central_speed * ratio_hi
    min_speed = max(bulk_v - SPEED_HALF_WIDTH_VTH * sigma, passband_lo)
    max_speed = min(bulk_v + SPEED_HALF_WIDTH_VTH * sigma, passband_hi)
    if max_speed <= min_speed:
        return True, None

    half = 0.5 * (max_speed - min_speed)
    mid = 0.5 * (max_speed + min_speed)
    points = mid + half * _GL_NODES_SPEED
    weights = half * _GL_WEIGHTS_SPEED

    speed_cubed_times_passband = np.empty_like(points)
    for i, v in enumerate(points):
        speed_cubed_times_passband[i] = v**3 * interpolate_passband(
            passband, elevation, v / central_speed
        )

    return False, SpeedQuadrature(points, weights, speed_cubed_times_passband)


@numba.njit
def _angular_limits(
    sw_params: SolarWindParams,
    rotation_matrix,
    region: Region,
    response_grid: ResponseGrid,
):
    half_width = _maxwellian_angular_extent(
        sw_params, response_grid.central_speed, EPSILON
    )
    bulk_az, bulk_el = bulk_angles_in_instrument_frame(sw_params, rotation_matrix)

    if region.is_sunglasses:
        el_lo, el_hi = response_grid.sg_passband.elevation_range
        az_lo, az_hi = -20.0, 20.0
    else:
        el_lo, el_hi = response_grid.oa_passband.elevation_range
        az_lo, az_hi = 20.0, 150.0
        if region.azimuth_sign < 0:
            az_lo, az_hi = -az_hi, -az_lo

    min_el, max_el = _clamp_window(bulk_el, half_width, el_lo, el_hi)
    min_az, max_az = _clamp_window(bulk_az, half_width, az_lo, az_hi)
    return min_el, max_el, min_az, max_az


@numba.njit
def _maxwellian_angular_extent(sw_params, central_speed, epsilon):
    sigma = thermal_speed(sw_params)
    speed = bulk_speed(sw_params)
    cos_theta = sigma**2 * np.log(epsilon) / (central_speed * speed) + 1
    return float(np.degrees(np.arccos(_clamp(cos_theta, -1, +1))))


@numba.njit
def _clamp(x: float, lower: float, upper: float) -> float:
    return min(max(x, lower), upper)


@numba.njit
def _clamp_window(
    center: float, half_width: float, lower_bound: float, upper_bound: float
):
    return (
        _clamp(center - half_width, lower_bound, upper_bound),
        _clamp(center + half_width, lower_bound, upper_bound),
    )
