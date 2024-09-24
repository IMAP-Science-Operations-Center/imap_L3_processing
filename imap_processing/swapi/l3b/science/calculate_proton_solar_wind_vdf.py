import numpy as np

from imap_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed_h_plus


def calculate_proton_solar_wind_vdf(energies, count_rates, efficiency, geometric_factor):
    velocities = calculate_sw_speed_h_plus(energies)

    proton_mass_per_charge = PROTON_MASS_KG / PROTON_CHARGE_COULOMBS
    numerator = 4 * np.pi * proton_mass_per_charge * count_rates
    denominator = (energies * geometric_factor * efficiency)
    probabilities = numerator / denominator

    return velocities, probabilities
