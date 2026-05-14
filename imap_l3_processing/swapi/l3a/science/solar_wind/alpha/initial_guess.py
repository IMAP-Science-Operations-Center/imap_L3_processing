from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike

from imap_l3_processing.constants import (
    ALPHA_PARTICLE_CHARGE_COULOMBS,
    ALPHA_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor


def esa_voltage_to_alpha_speed(esa_voltage: ArrayLike) -> np.ndarray:
    return (
        np.sqrt(
            2
            * SWAPI_K_FACTOR
            * ALPHA_PARTICLE_CHARGE_COULOMBS
            * np.abs(esa_voltage)
            / ALPHA_PARTICLE_MASS_KG
        )
        / METERS_PER_KILOMETER
    )


def get_alpha_peak_indices(residuals, energies, proton_peak_index) -> slice:
    energies = np.asarray(energies)
    assert np.all(energies[:-1] >= energies[1:]), "Energies must be decreasing"

    min_energy = 1.5 * energies[proton_peak_index]
    max_energy = 4.0 * energies[proton_peak_index]

    first_index_at_or_above_min_energy = None
    for i in reversed(range(proton_peak_index)):
        if energies[i] >= min_energy:
            first_index_at_or_above_min_energy = i
            break

    if first_index_at_or_above_min_energy is None:
        raise Exception("Proton peak too high to find alpha peak")

    window_energy_min = None
    for i in reversed(range(first_index_at_or_above_min_energy + 1)):
        is_increasing = residuals[i] > residuals[i + 1]
        if is_increasing:
            window_energy_min = energies[i]
            break

    if window_energy_min is None:
        raise Exception("Alpha peak not found")

    window_energy_max = max_energy

    masked_residuals = np.where(
        (energies >= window_energy_min) & (energies <= window_energy_max), residuals, 0
    )
    alpha_peak_index = masked_residuals.argmax()

    if alpha_peak_index < 3:
        raise Exception("Alpha peak too close to high-energy edge")
    return slice(alpha_peak_index - 3, alpha_peak_index + 2)


def calculate_initial_guess(
    alpha_ctx: SolarWindFitContext,
    proton_true_rate: ndarray,
    proton_temperature: float,
    proton_bulk_velocity_rtn: ndarray,
) -> Optional[tuple[float, float, float, ndarray]]:
    if alpha_ctx.count_rate.size == 0 or alpha_ctx.count_rate.ndim != 2:
        return None

    n_sweeps, n_bins = alpha_ctx.count_rate.shape
    counts_per_sweep = alpha_ctx.count_rate
    voltage_per_sweep = alpha_ctx.esa_voltage[0]
    proton_true_per_sweep = proton_true_rate.reshape(n_sweeps, n_bins)
    proton_obs_per_sweep = proton_true_per_sweep * deadtime_factor(
        proton_true_per_sweep
    )

    count_avg = counts_per_sweep.mean(axis=0)
    proton_bg_avg = proton_obs_per_sweep.mean(axis=0)
    energies_per_sweep = SWAPI_K_FACTOR * np.abs(voltage_per_sweep)
    proton_peak_index = np.argmax(proton_bg_avg)
    residual = np.maximum(0, count_avg - proton_bg_avg * 2)

    try:
        peak = get_alpha_peak_indices(residual, energies_per_sweep, proton_peak_index)
    except Exception:
        return None

    peak_idx = np.arange(peak.start, peak.stop)
    if len(peak_idx) < 3:
        return None
    residual_peak = np.maximum(residual[peak_idx], 0.0)
    if not np.any(residual_peak > 0):
        return None

    alpha_temperature = proton_temperature

    unit_alpha, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, proton_bulk_velocity_rtn, alpha_temperature, alpha_ctx.mass_kg),
        alpha_ctx,
    )
    unit_alpha_per_sweep = unit_alpha.reshape(n_sweeps, n_bins).mean(axis=0)
    denom = float(np.nanmean(unit_alpha_per_sweep[peak_idx]))
    if denom <= 0 or not np.isfinite(denom):
        return None
    alpha_density = float(np.nanmean(residual_peak)) / denom
    alpha_density = max(alpha_density, 1e-3)

    return (alpha_density, alpha_temperature, 0.0, peak_idx)
