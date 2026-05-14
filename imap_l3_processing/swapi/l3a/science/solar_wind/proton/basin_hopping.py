import numpy as np
from numpy import ndarray

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.response.deadtime import deadtime_factor


def flipped_seed(
    lm_result,
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
