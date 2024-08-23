import math

import numpy as np
import scipy
import uncertainties
from uncertainties import ufloat, wrap, umath
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG


def calculate_sw_speed_h_plus(energy):
    return umath.sqrt(2 * energy * PROTON_CHARGE_COULOMBS / PROTON_MASS_KG)


def get_peak_indices(count_rates, width) -> slice:
    max_indices = np.argwhere(count_rates == np.max(count_rates))
    left_min_index = np.min(max_indices)
    right_min_index = np.max(max_indices)

    if right_min_index - left_min_index > 1:
        raise Exception("Count rates contains multiple distinct peaks")

    return slice(left_min_index - width, right_min_index + width+1)


def find_peak_center_of_mass_index(peak_slice, count_rates):
    indices = np.arange(len(count_rates))
    peak_counts = count_rates[peak_slice]
    center_of_mass_index = np.sum(indices[peak_slice] * peak_counts) / np.sum(peak_counts)
    return center_of_mass_index


def sine_fit_function(spin_phase_angle, a, phi, b):
    return a * np.sin(np.deg2rad(spin_phase_angle + phi)) + b


def fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles):
    values, pcov = scipy.optimize.curve_fit(sine_fit_function, spin_phase_angles, nominal_values(centers_of_mass), sigma=std_devs(centers_of_mass),
                                         bounds=([0, 0, 0], [np.inf, 360, np.inf]),
                                            absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return uarray(values, perr)


def interpolate_energy(center_of_mass_index, energies):
    interpolate_lambda = lambda x: np.exp(np.interp(x, np.arange(len(energies)), np.log(energies)))
    interpolate = wrap(interpolate_lambda)
    # center = interpolate_lambda(center_of_mass_index.n)
    # left = interpolate_lambda(center_of_mass_index.n - center_of_mass_index.std_dev)
    # right = interpolate_lambda(center_of_mass_index.n + center_of_mass_index.std_dev)
    # print(left-center, center-right)
    # print((left-right)/2) # (right-center  + center-left)/2
    # print(interpolate(center_of_mass_index).s)
    # print("------")
    return interpolate(center_of_mass_index)


def times_for_sweep(start_time):
    time_step_per_bin = 12 / 72
    return start_time + time_step_per_bin * np.arange(72)


def extract_coarse_sweep(data: np.ndarray):
    if data.ndim > 1:
        return data[:, 1:63]
    else:
        return data[1:63]


def calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch):
    energies_at_center_of_mass = []
    energies_at_center_of_mass_uncertainties = []

    spin_angles_at_center_of_mass = []
    for i in range(len(epoch)):
        rates = coincidence_count_rates[i]
        angles = spin_angles[i]
        peak_slice = get_peak_indices(rates, 4)
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


def calculate_proton_solar_wind_speed(coincidence_count_rates, spin_angles, energies, epoch):
    energies = extract_coarse_sweep(energies)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)
    spin_angles = extract_coarse_sweep(spin_angles)

    energies_at_center_of_mass, spin_angles_at_center_of_mass = calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch)

    a, phi, b = fit_energy_per_charge_peak_variations(energies_at_center_of_mass, spin_angles_at_center_of_mass)

    proton_sw_speed = calculate_sw_speed_h_plus(b)
    return proton_sw_speed, a, phi, b

