import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import average_spin_axis_rtn
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
    optimize_solar_wind_params,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams


_MAX_BASIN_REFINE_ITERS = 6
_ROTATED_RMSE_RATIO_THRESHOLD = 10


def escape_local_minimum(
    first_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    spin_axis_rtn = average_spin_axis_rtn(ctx.rotation_matrices)

    current_result = first_result
    for _ in range(_MAX_BASIN_REFINE_ITERS):
        flipped_mse, flipped_seed_params = _flipped_seed(
            current_result, ctx, spin_axis_rtn
        )

        flipped_seed_is_substantially_worse_than_current = (
            flipped_mse >= current_result.mse * _ROTATED_RMSE_RATIO_THRESHOLD**2
        )
        if flipped_seed_is_substantially_worse_than_current:
            break

        flipped_result: OptimizeSolarWindParamsResult = _restart_from_flipped_params(
            flipped_seed_params, ctx
        )

        flipped_result_is_worse_than_current = flipped_result.mse > current_result.mse
        if flipped_result_is_worse_than_current:
            break

        current_result = flipped_result

    return current_result


def _flipped_seed(
    lm_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
    spin_axis_rtn: ndarray,
) -> tuple[float, SolarWindParams]:
    sw = lm_result.sw_params
    flipped_velocity = _flip_vector_about_axis(sw.bulk_velocity_rtn, spin_axis_rtn)
    unit_ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, flipped_velocity, sw.temperature, sw.mass),
        ctx,
    )
    count_rate_flat = ctx.count_rate.ravel()
    flipped_density = optimal_density_scale(unit_ideal_rates, count_rate_flat)

    # Same forward model as LM-1's residuals, evaluated self-consistently at
    # `flipped_density` so the basin-check ratio compares apples to apples.
    true_rate = flipped_density * unit_ideal_rates
    obs_pred = true_rate * deadtime_factor(true_rate)
    flipped_mse = float(np.mean((obs_pred - count_rate_flat) ** 2))

    return flipped_mse, SolarWindParams(
        density=flipped_density,
        bulk_velocity_rtn=flipped_velocity,
        temperature=sw.temperature,
        mass=sw.mass,
    )


def _flip_vector_about_axis(v: ndarray, axis: ndarray) -> ndarray:
    return 2.0 * axis * float(np.dot(axis, v)) - v


def _restart_from_flipped_params(
    flipped_params: SolarWindParams, ctx: SolarWindFitContext
) -> OptimizeSolarWindParamsResult:
    return optimize_solar_wind_params(flipped_params, ctx=ctx)
