from dataclasses import dataclass
import matplotlib.pyplot as plt

import numpy as np
import scipy
from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    coincidence_uncertainty: np.ndarray[float]

def calculate_sw_speed_h_plus(energy):
    proton_charge = 1.602176634e-19
    proton_mass = 1.67262192595e-27
    return np.sqrt(2 * energy * proton_charge / proton_mass)

def get_peak_indices(count_rates, width):
    left_min_index = np.min(np.nanargmax(count_rates))
    right_min_index = np.max(np.nanargmax(count_rates))
    return left_min_index - width, right_min_index + width


def find_peak_center_of_mass_index(left_index, right_index, count_rates):
    indices = np.arange(len(count_rates))
    peak = slice(left_index, right_index+1)
    peak_counts = count_rates[peak]
    center_of_mass_index = np.sum(indices[peak] * peak_counts) / np.sum(peak_counts)
    return center_of_mass_index


def sine_fit_function(spin_phase_angle, a, phi, b):
    return a * np.sin(np.deg2rad(spin_phase_angle + phi)) + b


def fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles):
    values, _ = scipy.optimize.curve_fit(sine_fit_function, spin_phase_angles, centers_of_mass)
    return values


def interpolate_energy(center_of_mass_index, energies):
    return np.exp(np.interp(center_of_mass_index.nominal_value, np.arange(len(energies)), np.log(energies)))


def times_for_sweep(start_time):
    time_step_per_bin = 12/72
    return start_time + time_step_per_bin * np.arange(72)


def get_artificial_spin_phase(time):
    arbitrary_offset = 0.4
    rotation_time = 15 * 1_000_000_000
    rotations = arbitrary_offset + time/rotation_time
    fractional, integral = np.modf(rotations)
    return fractional*360


def get_center_of_mass_and_spin_angle_for_single_sweep(sweep_count_rates, sweep_start_time, energies):
    left, right = get_peak_indices(sweep_count_rates, 4)
    center_of_mass_index = find_peak_center_of_mass_index(left, right, sweep_count_rates)
    energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)
    sweep_times = times_for_sweep(sweep_start_time)
    time_of_peak = np.interp(center_of_mass_index.nominal_value, np.arange(len(sweep_times)), sweep_times)
    spin_phase_angle = get_artificial_spin_phase(time_of_peak)

    return np.array([energy_at_center_of_mass, spin_phase_angle])


def read_l2_data(cdf_path: str) -> SwapiL2Data:
    cdf = CDF(cdf_path)
    return SwapiL2Data(cdf.raw_var("epoch")[...], cdf["energy"][...], cdf["swp_coin_rate"][...], cdf["swp_coin_unc"][...])


fig, axes = plt.subplots(4)


def plot_sweeps(data):
    not_nan = data.energy > 0

    axes[0].loglog(data.energy[not_nan], data.coincidence_count_rate[:,not_nan][1,:], marker='.', linestyle="None")
    axes[0].set(xlabel="Energy", ylabel="Count Rate")
    axes[1].loglog(data.energy[not_nan], data.coincidence_count_rate[:, not_nan][1, :], marker='.', linestyle="None")
    axes[1].set(xlabel="Energy", ylabel="Count Rate")
    axes[2].loglog(data.energy[not_nan], data.coincidence_count_rate[:, not_nan][1, :], marker='.', linestyle="None")
    axes[2].set(xlabel="Energy", ylabel="Count Rate")


def plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass):
    fit_xs = np.arange(0, 360, 3)
    fit_ys = sine_fit_function(fit_xs, a, phi % 360, b)
    axes[3].scatter(spin_angles, centers_of_mass)
    axes[3].plot(fit_xs, fit_ys)
    axes[3].set(xlabel="Phase Angle", ylabel="Energy")

def main():
    data = read_l2_data("test_data/imap_swapi_l2_sci-1min_20100101_v001_EDITED.cdf")

    coincidence_count_range_with_uncertainty = [uarray(data.coincidence_count_rate[i, :], data.coincidence_uncertainty[i, :]) for i in range(len(data.epoch))]

    plot_sweeps(data)

    center_of_mass_and_spin = np.array([get_center_of_mass_and_spin_angle_for_single_sweep(
        coincidence_count_range_with_uncertainty[i], data.epoch[i], data.energy) for i in range(len(data.epoch))])

    centers_of_mass = center_of_mass_and_spin[:, 0]
    spin_angles = center_of_mass_and_spin[:, 1]

    a, phi, b = fit_energy_per_charge_peak_variations(centers_of_mass, spin_angles)

    plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass)

    print(f"SW H+ speed: {calculate_sw_speed_h_plus(b)}")
    print(f"SW H+ clock angle: {phi % 360}")

    plt.show()


main()