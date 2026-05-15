import math

import numba
import numpy as np

from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    SolarWindParams,
    bulk_angles_in_instrument_frame,
    bulk_speed, thermal_speed,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import (
    count_rate_conversion_factor,
)
from imap_l3_processing.swapi.response.azimuthal_transmission import (
    interpolate_azimuthal_transmission,
)
from imap_l3_processing.swapi.response.passband_grid import (
    speed_ratio_range_at_elevation,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid


OA_SCAN_THRESHOLD = 1e-6
OA_SCAN_RESOLUTION = 64
OA_SKIP_FRACTION = 1e-3


@numba.njit
def trim_open_aperture(
    response_grid: ResponseGrid,
    sw_params: SolarWindParams,
    rotation_matrix,
    min_elevation: float,
    max_elevation: float,
    azimuth_lo: float,
    azimuth_hi: float,
    sg_rate: float,
):
    if azimuth_hi <= azimuth_lo:
        return 0.0, 0.0

    _, bulk_el = bulk_angles_in_instrument_frame(sw_params, rotation_matrix)
    scan_elevation = min(max(bulk_el, min_elevation), max_elevation)
    scan_azimuths = np.linspace(azimuth_lo, azimuth_hi, OA_SCAN_RESOLUTION)
    transmission_x_maxwellian = _evaluate_oa_integrand_along_azimuth(
        response_grid, sw_params, rotation_matrix, scan_azimuths, scan_elevation
    )

    threshold = OA_SCAN_THRESHOLD * np.max(transmission_x_maxwellian)
    n = transmission_x_maxwellian.shape[0]
    lower_index = 0
    for i in range(n):
        if transmission_x_maxwellian[i] > threshold:
            lower_index = max(i - 1, 0)
            break
    upper_index = n - 1
    for i in range(n - 1, -1, -1):
        if transmission_x_maxwellian[i] > threshold:
            upper_index = min(i + 1, n - 1)
            break

    dphi_deg = scan_azimuths[1] - scan_azimuths[0]
    transmission_az_integral = float(np.trapezoid(
        transmission_x_maxwellian[lower_index : upper_index + 1], dx=dphi_deg
    ))
    upper_bound = _oa_rate_upper_bound(
        response_grid,
        sw_params,
        transmission_az_integral,
        min_elevation,
        max_elevation,
    )
    if upper_bound < max(0.1, OA_SKIP_FRACTION * sg_rate):
        return 0.0, 0.0
    return scan_azimuths[lower_index], scan_azimuths[upper_index]


@numba.njit
def _evaluate_oa_integrand_along_azimuth(
    response_grid: ResponseGrid,
    sw_params: SolarWindParams,
    rotation_matrix,
    scan_azimuths: np.ndarray,
    scan_elevation: float,
) -> np.ndarray:
    sigma = thermal_speed(sw_params)
    speed = bulk_speed(sw_params)
    central_speed = response_grid.central_speed
    bulk_az, bulk_el = bulk_angles_in_instrument_frame(sw_params, rotation_matrix)

    sin_bulk_el = math.sin(np.radians(bulk_el))
    cos_bulk_el = math.cos(np.radians(bulk_el))
    sin_scan_el = math.sin(np.radians(scan_elevation))
    cos_scan_el = math.cos(np.radians(scan_elevation))
    cos_view_to_bulk = (
        sin_bulk_el * sin_scan_el + cos_bulk_el * cos_scan_el
        * np.cos(np.radians(scan_azimuths - bulk_az))
    )
    delta_v_sq = (
        central_speed**2 + speed**2 - 2.0 * central_speed * speed * cos_view_to_bulk
    )
    maxwellian = np.exp(-delta_v_sq / (2.0 * sigma**2))

    for i in range(scan_azimuths.shape[0]):
        maxwellian[i] *= interpolate_azimuthal_transmission(
            response_grid.azimuthal_transmission,
            scan_azimuths[i],
        )

    return maxwellian


@numba.njit
def _oa_rate_upper_bound(
    response_grid: ResponseGrid,
    sw_params: SolarWindParams,
    transmission_az_integral: float,
    min_elevation: float,
    max_elevation: float,
) -> float:
    central_speed = response_grid.central_speed
    delta_theta_deg = max_elevation - min_elevation
    ratio_lo, ratio_hi = speed_ratio_range_at_elevation(response_grid.oa_passband, 0.0)
    delta_v = central_speed * (ratio_hi - ratio_lo)
    return (
            count_rate_conversion_factor(sw_params, response_grid)
            * central_speed ** 3
            * delta_theta_deg
            * delta_v
            * transmission_az_integral
    )
