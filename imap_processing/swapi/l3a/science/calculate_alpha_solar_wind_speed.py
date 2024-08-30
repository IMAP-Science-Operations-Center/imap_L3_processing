import numpy as np
from uncertainties import umath

from imap_processing.constants import ALPHA_PARTICLE_CHARGE_COULOMBS, ALPHA_PARTICLE_MASS_KG
from imap_processing.swapi.l3a.science.speed_calculation import get_peak_indices, interpolate_energy, \
    find_peak_center_of_mass_index, extract_coarse_sweep


def get_alpha_peak_indices(count_rates, energies) -> slice:
    energies = np.asarray(energies)
    assert np.all(energies[:-1] >= energies[1:]), "Energies must be decreasing"
    proton_peak_index = get_peak_indices(count_rates, 0).start

    def find_start_of_alpha_particle_peak():
        last_count_rate = count_rates[proton_peak_index]
        for i in reversed(range(proton_peak_index - 1)):
            if count_rates[i] > last_count_rate:
                return i
            last_count_rate = count_rates[i]
    start_of_alpha_peak = find_start_of_alpha_particle_peak()
    if start_of_alpha_peak is None:
        raise Exception("Alpha peak not found")
    indices = np.arange(len(count_rates))
    after_proton_peak = indices <= start_of_alpha_peak
    before_4x_proton_energy = energies <= 4 * energies[proton_peak_index]
    mask = after_proton_peak & before_4x_proton_energy
    return get_peak_indices(count_rates, 2, mask)

def calculate_alpha_center_of_mass(coincidence_count_rates, energies):
    alpha_particle_peak_slice = get_alpha_peak_indices(coincidence_count_rates, energies)
    center_of_mass_index = find_peak_center_of_mass_index(alpha_particle_peak_slice, coincidence_count_rates)
    energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)

    return energy_at_center_of_mass

def calculate_sw_speed_alpha(energy):
    return umath.sqrt(2 * energy * ALPHA_PARTICLE_CHARGE_COULOMBS / ALPHA_PARTICLE_MASS_KG)

def calculate_alpha_solar_wind_speed(coincidence_count_rates, energies):
    energies = extract_coarse_sweep(energies)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)

    average_coin_rates = np.sum(coincidence_count_rates, axis=0) / len(coincidence_count_rates)

    energy_at_center_of_mass = calculate_alpha_center_of_mass(average_coin_rates, energies)
    speed = calculate_sw_speed_alpha(energy_at_center_of_mass)
    return speed