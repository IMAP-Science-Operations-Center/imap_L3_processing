"""Numba reference for the SWAPI proton solar wind integral with fixed limits.

Mirrors the physics in `imap_l3_processing/swapi/l3a/science/solar_wind/forward_model.py::calculate_integral`
but on a fixed dense trapezoid grid (no dynamic angular/speed windowing). Used
as a ground-truth reference for the production GL quadrature in
`docs/swapi/figure_src/plot_validation_scatter.py` and `plot_spectra.py`.

Fixed integration grid:
  elevation:  -15 to 15 deg @ 0.05° (601 pts)
  azimuth SG: -20 to 20 deg @ 0.05° (801 pts)
  azimuth OA: 0.05° in transition |az| ∈ [20, 30], 0.5° to ±150° (441 pts/side)
              — split at the |az| < 20° dead band so trapezoid doesn't
                bridge the gap.
  speed:      200 samples from 0.9 to 1.1 × central_speed

The 0.05° elevation/SG-azimuth spacing was chosen to match the bilinear-
interpolated integrand: cold plasma (vth ~10 km/s, σ_el = vth/v_b ≈ 0.4°) at
SG passband elevation edges produces a near-cliff in the integrand, where
0.1° trap undersampled the second derivative and biased the integral low by
~1.4% (worst-case at bulk_el just outside SG el range).
"""

import math

import numpy as np
from numba import njit, prange

from imap_l3_processing.swapi.l3a.science.solar_wind import params
from imap_l3_processing.swapi.l3a.science.solar_wind.params import bulk_speed
from imap_l3_processing.swapi.response.swapi_response import ResponseGrid

_EL = np.linspace(-15.0, 15.0, 601)
_AZ_SG = np.linspace(-20.0, 20.0, 801)
_AZ_OA_NEG = np.concatenate(
    [np.arange(-150.0, -30.0, 0.5), np.linspace(-30.0, -20.0, 201)]
)
_AZ_OA_POS = np.concatenate([np.linspace(20.0, 30.0, 201), np.arange(30.5, 151.0, 0.5)])
_SP_RATIO = np.linspace(0.9, 1.1, 200)


def _trap_weights(x):
    """Per-point trapezoid weights on a 1-D grid (handles non-uniform spacing)."""
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


_EL_W = _trap_weights(_EL)
_AZ_SG_W = _trap_weights(_AZ_SG)
_AZ_OA_NEG_W = _trap_weights(_AZ_OA_NEG)
_AZ_OA_POS_W = _trap_weights(_AZ_OA_POS)
_SP_RATIO_W = _trap_weights(_SP_RATIO)


@njit(fastmath=True, inline="always")
def _interpolate_passband(values, min_el, el_sp, min_r, r_sp, el, speed_ratio):
    """Bilinear on the (elevation, speed_ratio) passband grid; mirrors
    `passband_grid._bilinear_interpolate`."""
    i = (el - min_el) / el_sp
    j = (speed_ratio - min_r) / r_sp
    if i < 0 or i + 1 >= values.shape[0] or j < 0 or j + 1 >= values.shape[1]:
        return 0.0
    i0, j0 = int(i), int(j)
    wi, wj = i - i0, j - j0
    return (1 - wi) * ((1 - wj) * values[i0, j0] + wj * values[i0, j0 + 1]) + wi * (
        (1 - wj) * values[i0 + 1, j0] + wj * values[i0 + 1, j0 + 1]
    )


@njit(fastmath=True, inline="always")
def _interpolate_transmission(transmission, transmission_spacing, az):
    """Mirror of `azimuthal_transmission.interpolate_azimuthal_transmission`."""
    az = (az + 180.0) % 360.0 - 180.0
    f = abs(az) / transmission_spacing
    n = transmission.shape[0]
    i0 = int(f)
    if i0 < 0:
        i0 = 0
    elif i0 >= n:
        i0 = n - 1
    i1 = i0 + 1 if i0 + 1 < n else n - 1
    return transmission[i0] * (i1 - f) + transmission[i1] * (f - i0)


@njit(fastmath=True)
def _integrate_region(
    passband_values,
    rotation,
    bulk_r,
    bulk_t,
    bulk_n,
    bulk_speed_value,
    inv_two_sigma_sq,
    central_speed,
    transmission,
    transmission_spacing,
    pb_min_el,
    pb_el_sp,
    pb_min_r,
    pb_r_sp,
    el_points,
    el_weights,
    az_points,
    az_weights,
    sp_ratio_points,
    sp_ratio_weights,
):
    bulk_speed_squared = bulk_speed_value * bulk_speed_value
    total = 0.0
    for i_el in range(el_points.shape[0]):
        el = el_points[i_el]
        sin_el = math.sin(math.radians(el))
        cos_el = math.cos(math.radians(el))
        el_weight = el_weights[i_el] * cos_el
        for i_az in range(az_points.shape[0]):
            az = az_points[i_az]
            sin_az = math.sin(math.radians(az))
            cos_az = math.cos(math.radians(az))
            direction_x = -cos_el * sin_az
            direction_y = -cos_el * cos_az
            direction_z = -sin_el
            direction_r = (
                rotation[0, 0] * direction_x
                + rotation[0, 1] * direction_y
                + rotation[0, 2] * direction_z
            )
            direction_t = (
                rotation[1, 0] * direction_x
                + rotation[1, 1] * direction_y
                + rotation[1, 2] * direction_z
            )
            direction_n = (
                rotation[2, 0] * direction_x
                + rotation[2, 1] * direction_y
                + rotation[2, 2] * direction_z
            )
            bulk_along_direction = (
                direction_r * bulk_r + direction_t * bulk_t + direction_n * bulk_n
            )
            transmission_at_az = _interpolate_transmission(
                transmission, transmission_spacing, az
            )
            az_weight = az_weights[i_az] * transmission_at_az
            for i_sp in range(sp_ratio_points.shape[0]):
                speed_ratio = sp_ratio_points[i_sp]
                speed = central_speed * speed_ratio
                passband = _interpolate_passband(
                    passband_values,
                    pb_min_el,
                    pb_el_sp,
                    pb_min_r,
                    pb_r_sp,
                    el,
                    speed_ratio,
                )
                exponent = (
                    speed * speed
                    + bulk_speed_squared
                    - 2.0 * speed * bulk_along_direction
                ) * inv_two_sigma_sq
                integrand = passband * speed**3 * math.exp(-exponent)
                speed_weight = central_speed * sp_ratio_weights[i_sp]
                total += el_weight * az_weight * speed_weight * integrand
    return total


@njit(parallel=True, fastmath=True)
def _batch(
    sg_values_all,
    oa_values_all,
    rotations,
    bulk_velocities_rtn,
    bulk_speeds,
    thermal_speeds,
    densities,
    central_speeds,
    central_effective_areas,
    transmission,
    transmission_spacing,
    pb_min_el,
    pb_el_sp,
    pb_min_r,
    pb_r_sp,
    el_points,
    el_weights,
    az_sg_points,
    az_sg_weights,
    az_oa_neg_points,
    az_oa_neg_weights,
    az_oa_pos_points,
    az_oa_pos_weights,
    sp_ratio_points,
    sp_ratio_weights,
):
    n = sg_values_all.shape[0]
    out = np.empty(n)
    for i in prange(n):
        sigma = thermal_speeds[i]
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        bulk_r = bulk_velocities_rtn[i, 0]
        bulk_t = bulk_velocities_rtn[i, 1]
        bulk_n = bulk_velocities_rtn[i, 2]
        bulk_speed_value = bulk_speeds[i]
        rotation = rotations[i]
        central_speed = central_speeds[i]

        sg_total = _integrate_region(
            sg_values_all[i],
            rotation,
            bulk_r,
            bulk_t,
            bulk_n,
            bulk_speed_value,
            inv_two_sigma_sq,
            central_speed,
            transmission,
            transmission_spacing,
            pb_min_el,
            pb_el_sp,
            pb_min_r,
            pb_r_sp,
            el_points,
            el_weights,
            az_sg_points,
            az_sg_weights,
            sp_ratio_points,
            sp_ratio_weights,
        )
        oa_neg_total = _integrate_region(
            oa_values_all[i],
            rotation,
            bulk_r,
            bulk_t,
            bulk_n,
            bulk_speed_value,
            inv_two_sigma_sq,
            central_speed,
            transmission,
            transmission_spacing,
            pb_min_el,
            pb_el_sp,
            pb_min_r,
            pb_r_sp,
            el_points,
            el_weights,
            az_oa_neg_points,
            az_oa_neg_weights,
            sp_ratio_points,
            sp_ratio_weights,
        )
        oa_pos_total = _integrate_region(
            oa_values_all[i],
            rotation,
            bulk_r,
            bulk_t,
            bulk_n,
            bulk_speed_value,
            inv_two_sigma_sq,
            central_speed,
            transmission,
            transmission_spacing,
            pb_min_el,
            pb_el_sp,
            pb_min_r,
            pb_r_sp,
            el_points,
            el_weights,
            az_oa_pos_points,
            az_oa_pos_weights,
            sp_ratio_points,
            sp_ratio_weights,
        )
        rate = sg_total + oa_neg_total + oa_pos_total
        # Matches `count_rate_conversion_factor` in solar_wind/utils.py.
        prefactor = (
            central_effective_areas[i]
            * densities[i]
            * (math.sqrt(2.0 * math.pi) * sigma) ** -3
            * (math.pi / 180.0) ** 2
            * 1e5
        )
        out[i] = rate * prefactor
    return out


def reference_integrals_batch(response_grids, sws, rotation_matrices):
    """Compute N reference integrals in parallel via numba JIT.

    Each `response_grids[i]` carries its sample's SG/OA passbands, central speed,
    central effective area, and (the shared) azimuthal-transmission table —
    matching the signature of `calculate_integral`. `rotation_matrices` (shape
    (N, 3, 3)) map SWAPI XYZ → RTN per sample.
    """
    n = len(sws)
    if n == 0:
        return np.empty(0)
    first_grid = response_grids[0]
    first_sg = first_grid.sg_passband
    nel, nsp = first_sg.values.shape
    sg_values_all = np.empty((n, nel, nsp))
    oa_values_all = np.empty((n, nel, nsp))
    rotations = np.ascontiguousarray(rotation_matrices, dtype=np.float64)
    bulk_velocities_rtn = np.empty((n, 3))
    bulk_speeds = np.empty(n)
    thermal_speeds = np.empty(n)
    densities = np.empty(n)
    central_speeds = np.empty(n)
    central_effective_areas = np.empty(n)
    for i in range(n):
        rg = response_grids[i]
        sg_values_all[i] = rg.sg_passband.values
        oa_values_all[i] = rg.oa_passband.values
        central_speeds[i] = rg.central_speed
        central_effective_areas[i] = rg.central_effective_area
        bulk_velocities_rtn[i] = sws[i].velocity_rtn
        bulk_speeds[i] = bulk_speed(sws[i])
        thermal_speeds[i] = params.thermal_speed(sws[i])
        densities[i] = sws[i].density
    transmission = np.ascontiguousarray(
        first_grid.azimuthal_transmission.values, dtype=np.float64
    )
    return _batch(
        sg_values_all,
        oa_values_all,
        rotations,
        bulk_velocities_rtn,
        bulk_speeds,
        thermal_speeds,
        densities,
        central_speeds,
        central_effective_areas,
        transmission,
        float(first_grid.azimuthal_transmission.spacing),
        float(first_sg.min_elevation),
        float(first_sg.elevation_spacing),
        float(first_sg.min_speed_ratio),
        float(first_sg.speed_ratio_spacing),
        _EL,
        _EL_W,
        _AZ_SG,
        _AZ_SG_W,
        _AZ_OA_NEG,
        _AZ_OA_NEG_W,
        _AZ_OA_POS,
        _AZ_OA_POS_W,
        _SP_RATIO,
        _SP_RATIO_W,
    )


def reference_integral_fixed_limits(
    response_grid: ResponseGrid, sw, rotation_matrix
) -> float:
    """Single-sample ground truth (delegates to the parallel batch)."""
    rm = np.asarray(rotation_matrix, dtype=np.float64).reshape(1, 3, 3)
    return float(reference_integrals_batch([response_grid], [sw], rm)[0])
