import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import uncertainties
from uncertainties import ufloat, wrap, umath
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, ALPHA_PARTICLE_MASS_KG, \
    ALPHA_PARTICLE_CHARGE_COULOMBS
from imap_processing.swapi.l3a.science.speed_calculation import get_peak_indices, find_peak_center_of_mass_index, \
    interpolate_energy, extract_coarse_sweep


def sine_fit_function(spin_phase_angle, a, phi, b):
    return a * np.sin(np.deg2rad(spin_phase_angle + phi)) + b


def fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles):
    values, pcov = scipy.optimize.curve_fit(sine_fit_function, spin_phase_angles, nominal_values(centers_of_mass), sigma=std_devs(centers_of_mass),
                                         bounds=([0, 0, 0], [np.inf, 360, np.inf]),
                                            absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return uarray(values, perr)

def get_proton_peak_indices(count_rates):
    return get_peak_indices(count_rates, 4)

def calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch):
    energies_at_center_of_mass = []
    energies_at_center_of_mass_uncertainties = []

    spin_angles_at_center_of_mass = []
    for i in range(len(epoch)):
        rates = coincidence_count_rates[i]
        angles = spin_angles[i]
        peak_slice = get_proton_peak_indices(rates)
        center_of_mass_index = find_peak_center_of_mass_index(peak_slice, rates)
        energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)
        """
            If the L2 cdf does not provide the spin angle, this will have to be retrieved from SPICE. Otherwise, we can
            interpolate over the spin angles directly from the L2 data
        
            sweep_times = times_for_sweep(sweep_start_time)
            time_of_peak = np.interp(center_of_mass_index, np.arange(len(sweep_times)), sweep_times)
            spin_phase_angle = get_spin_phase_from_spice(time_of_peak)
            """
        spin_angle = np.interp(center_of_mass_index.nominal_value, np.arange(len(angles)), angles)
        energies_at_center_of_mass.append(energy_at_center_of_mass.nominal_value)
        energies_at_center_of_mass_uncertainties.append(energy_at_center_of_mass.std_dev)
        spin_angles_at_center_of_mass.append(spin_angle)

    return uarray(energies_at_center_of_mass, energies_at_center_of_mass_uncertainties), spin_angles_at_center_of_mass


def calculate_sw_speed_h_plus(energy):
    return umath.sqrt(2 * energy * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)

def calculate_proton_solar_wind_speed(coincidence_count_rates, spin_angles, energies, epoch):
    energies = extract_coarse_sweep(energies)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)
    spin_angles = extract_coarse_sweep(spin_angles)

    energies_at_center_of_mass, spin_angles_at_center_of_mass = calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch)

    a, phi, b = fit_energy_per_charge_peak_variations(energies_at_center_of_mass, spin_angles_at_center_of_mass)

    proton_sw_speed = calculate_sw_speed_h_plus(b)
    return proton_sw_speed, a, phi, b



