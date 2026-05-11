import numpy as np
import scipy.optimize
from numpy import ndarray
from numpy.typing import ArrayLike

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    METERS_PER_KILOMETER,
    PROTON_CHARGE_COULOMBS,
    PROTON_MASS_KG,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.utils import optimal_density_scale
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import average_spin_axis_rtn
from imap_l3_processing.swapi.l3a.science.solar_wind.state import SolarWindParams, temperature_to_thermal_speed, \
    thermal_speed_to_temperature


# 1 eV
INITIAL_TEMPERATURE_FLOOR_K = (
    PROTON_CHARGE_COULOMBS / BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
)


def _esa_voltage_to_proton_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * PROTON_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / PROTON_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def calculate_initial_guess(ctx: SolarWindFitContext) -> SolarWindParams:
    speed = _esa_voltage_to_proton_speed(ctx.esa_voltage)

    peak_idx = np.nanargmax(ctx.count_rate)
    bulk_speed_seed = float(speed[peak_idx])

    temperature_seed = max(
        60000.0 * (bulk_speed_seed / 400.0) ** 2,
        INITIAL_TEMPERATURE_FLOOR_K,
    )

    bulk_speed_init, temperature = _gaussian_refine_bulk_speed_and_temperature(
        speed,
        ctx.count_rate,
        bulk_speed_seed,
        temperature_seed,
        ctx.mass_kg,
    )

    spin_axis_rtn = average_spin_axis_rtn(ctx.rotation_matrices)
    bulk_velocity_rtn = -bulk_speed_init * spin_axis_rtn

    unit_ideal_rates, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, bulk_velocity_rtn, temperature, ctx.mass_kg),
        ctx,
    )
    density = optimal_density_scale(unit_ideal_rates, ctx.count_rate)

    return SolarWindParams(
        density=density,
        bulk_velocity_rtn=bulk_velocity_rtn,
        temperature=temperature,
        mass=ctx.mass_kg,
    )


def _gaussian_refine_bulk_speed_and_temperature(
    speed: ndarray,
    count_rate: ndarray,
    bulk_speed_seed: float,
    temperature_seed: float,
    mass_kg: float,
) -> tuple[float, float]:
    def gaussian(v, amplitude, mean, sigma):
        return amplitude * np.exp(-0.5 * ((v - mean) / sigma) ** 2)

    seed_normalization_coefficient = float(np.nanmax(count_rate))
    seed_center = bulk_speed_seed
    seed_scale = temperature_to_thermal_speed(mass_kg, temperature_seed)
    p0 = [seed_normalization_coefficient, seed_center, seed_scale]

    try:
        (_, bulk_speed_fit, sigma_fit), _ = scipy.optimize.curve_fit(
            gaussian,
            speed,
            count_rate,
            p0=p0,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Initial-guess Gaussian fit failed for spectrum with peak-bin "
            f"speed {bulk_speed_seed:.1f} km/s and seed temperature "
            f"{temperature_seed:.0f} K."
        ) from e

    sigma_fit = abs(float(sigma_fit))
    temperature_fit = max(
        thermal_speed_to_temperature(sigma_fit, mass_kg),
        INITIAL_TEMPERATURE_FLOOR_K,
    )
    return float(bulk_speed_fit), float(temperature_fit)


