"""Solar-wind forward model and analytic Jacobian.

Derivation in `docs/swapi/solar-wind-moments.md`."""

import math

import numba
import numpy as np

from imap_l3_processing.swapi.l3a.science.solar_wind.azimuthal_regions import (
    REGION_OPEN_APERTURE_NEG,
    REGION_OPEN_APERTURE_POS,
    REGION_SUNGLASSES,
    Region,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.integration_limits import (
    AngularQuadrature,
    get_angular_quadrature,
    get_speed_quadrature,
    speed_window_misses_passband,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import (
    count_rate_conversion_factor,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    N_STATE,
    SolarWindParams,
    VELOCITY_SLICE,
    bulk_speed,
    thermal_speed,
)
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid


@numba.njit
def model_solar_wind_ideal_coincidence_rates(
    sw_params: SolarWindParams,
    ctx: SolarWindFitContext,
):
    n = len(ctx.response_grids)
    rates = np.empty(n)
    jacobian = np.empty((n, N_STATE))
    for i in range(n):
        rates[i], jacobian[i] = calculate_integral(
            sw_params, ctx.response_grids[i], ctx.rotation_matrices[i]
        )
    return rates, jacobian


@numba.njit(fastmath=True)
def calculate_integral(
    sw_params: SolarWindParams,
    response_grid: ResponseGrid,
    rotation_matrix,
):
    jacobian_row = np.zeros(N_STATE)
    if speed_window_misses_passband(sw_params, response_grid):
        return 0.0, jacobian_row

    rate = 0.0
    sg_rate = 0.0
    for region in [
        REGION_SUNGLASSES,
        REGION_OPEN_APERTURE_NEG,
        REGION_OPEN_APERTURE_POS,
    ]:
        skip_region, angular_quadrature = get_angular_quadrature(
            sw_params, response_grid, region, rotation_matrix, sg_rate
        )
        if not skip_region:
            region_rate, region_jac = _integrate_region(
                sw_params,
                response_grid,
                region,
                angular_quadrature,
                rotation_matrix,
            )
            rate += region_rate
            jacobian_row += region_jac
            if region.is_sunglasses:
                sg_rate = region_rate
    return rate, jacobian_row


@numba.njit(fastmath=True)
def _integrate_region(
    sw_params: SolarWindParams,
    response_grid: ResponseGrid,
    region: Region,
    angular_quadrature: AngularQuadrature,
    rotation_xyz_to_rtn,
):
    bulk_velocity = sw_params.bulk_velocity_rtn
    bulk_r = bulk_velocity[0]
    bulk_t = bulk_velocity[1]
    bulk_n = bulk_velocity[2]

    elevation_rate = 0.0
    elevation_log_temperature_jacobian = 0.0
    elevation_jacobian_r = 0.0
    elevation_jacobian_t = 0.0
    elevation_jacobian_n = 0.0
    for i_elevation in range(angular_quadrature.elevation_points.shape[0]):
        skip_elevation, speed_quadrature = get_speed_quadrature(
            sw_params,
            response_grid,
            region,
            angular_quadrature.elevation_points[i_elevation],
        )
        if skip_elevation:
            continue

        azimuth_rate = 0.0
        azimuth_log_temperature_jacobian = 0.0
        azimuth_jacobian_r = 0.0
        azimuth_jacobian_t = 0.0
        azimuth_jacobian_n = 0.0
        for i_azimuth in range(angular_quadrature.azimuth_points.shape[0]):
            direction_r, direction_t, direction_n = _direction_in_rtn(
                angular_quadrature, rotation_xyz_to_rtn, i_elevation, i_azimuth
            )

            bulk_velocity_along_direction = (
                direction_r * bulk_r + direction_t * bulk_t + direction_n * bulk_n
            )

            rate_integral = 0.0
            log_temperature_jacobian_integral = 0.0
            velocity_along_direction_integral = 0.0
            for i_speed in range(speed_quadrature.points.shape[0]):
                speed = speed_quadrature.points[i_speed]
                exponent = (
                    speed ** 2
                    + bulk_speed(sw_params) ** 2
                    - 2 * speed * bulk_velocity_along_direction
                ) * 1.0 / (2 * thermal_speed(sw_params) ** 2)
                weighted_integrand = (
                    speed_quadrature.weights[i_speed]
                    * speed_quadrature.speed_cubed_times_passband[i_speed]
                    * math.exp(-exponent)
                )
                rate_integral += weighted_integrand

                log_temperature_jacobian_integral += weighted_integrand * exponent
                velocity_along_direction_integral += weighted_integrand * speed

            azimuth_weight = (
                angular_quadrature.azimuth_weights[i_azimuth]
                * angular_quadrature.transmission_azimuth[i_azimuth]
            )

            azimuth_rate += azimuth_weight * rate_integral

            azimuth_log_temperature_jacobian += (
                azimuth_weight * log_temperature_jacobian_integral
            )
            azimuth_jacobian_r += azimuth_weight * (
                direction_r * velocity_along_direction_integral - bulk_r * rate_integral
            )
            azimuth_jacobian_t += azimuth_weight * (
                direction_t * velocity_along_direction_integral - bulk_t * rate_integral
            )
            azimuth_jacobian_n += azimuth_weight * (
                direction_n * velocity_along_direction_integral - bulk_n * rate_integral
            )

        elevation_weight = (
            angular_quadrature.elevation_weights[i_elevation]
            * angular_quadrature.cos_elevation[i_elevation]
        )
        elevation_rate += azimuth_rate * elevation_weight
        elevation_log_temperature_jacobian += (
            azimuth_log_temperature_jacobian * elevation_weight
        )
        elevation_jacobian_r += azimuth_jacobian_r * elevation_weight
        elevation_jacobian_t += azimuth_jacobian_t * elevation_weight
        elevation_jacobian_n += azimuth_jacobian_n * elevation_weight

    count_rate_factor = count_rate_conversion_factor(sw_params, response_grid)
    rate = elevation_rate * count_rate_factor

    jacobian = np.empty(N_STATE)
    jacobian[LOG_DENSITY_IDX] = rate
    jacobian[LOG_TEMPERATURE_IDX] = (
        elevation_log_temperature_jacobian * count_rate_factor - 1.5 * rate
    )
    jacobian[VELOCITY_SLICE.start] = (
            elevation_jacobian_r * count_rate_factor / thermal_speed(sw_params) ** 2
    )
    jacobian[VELOCITY_SLICE.start + 1] = (
            elevation_jacobian_t * count_rate_factor / thermal_speed(sw_params) ** 2
    )
    jacobian[VELOCITY_SLICE.start + 2] = (
            elevation_jacobian_n * count_rate_factor / thermal_speed(sw_params) ** 2
    )

    return rate, jacobian


@numba.njit(fastmath=True)
def _direction_in_rtn(angular_quadrature, rotation_xyz_to_rtn, i_elevation, i_azimuth):
    cos_elevation = angular_quadrature.cos_elevation[i_elevation]
    sin_elevation = angular_quadrature.sin_elevation[i_elevation]
    sin_azimuth = angular_quadrature.sin_azimuth[i_azimuth]
    cos_azimuth = angular_quadrature.cos_azimuth[i_azimuth]
    direction_x = -cos_elevation * sin_azimuth
    direction_y = -cos_elevation * cos_azimuth
    direction_z = -sin_elevation
    return (
        rotation_xyz_to_rtn[0, 0] * direction_x
        + rotation_xyz_to_rtn[0, 1] * direction_y
        + rotation_xyz_to_rtn[0, 2] * direction_z,
        rotation_xyz_to_rtn[1, 0] * direction_x
        + rotation_xyz_to_rtn[1, 1] * direction_y
        + rotation_xyz_to_rtn[1, 2] * direction_z,
        rotation_xyz_to_rtn[2, 0] * direction_x
        + rotation_xyz_to_rtn[2, 1] * direction_y
        + rotation_xyz_to_rtn[2, 2] * direction_z,
    )

