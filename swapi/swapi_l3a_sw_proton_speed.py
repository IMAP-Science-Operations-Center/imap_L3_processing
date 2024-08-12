import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt

import numpy as np
import scipy
from matplotlib.figure import Figure
from spacepy.pycdf import CDF


@dataclass
class SwapiL2Data:
    epoch: np.ndarray[float]
    energy: np.ndarray[float]
    coincidence_count_rate: np.ndarray[float]
    spin_angles: np.ndarray[float]  # not currently in the L2 cdf, is in the sample data provided by Bishwas


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
    peak = slice(left_index, right_index + 1)
    peak_counts = count_rates[peak]
    center_of_mass_index = np.sum(indices[peak] * peak_counts) / np.sum(peak_counts)
    return center_of_mass_index


def sine_fit_function(spin_phase_angle, a, phi, b):
    return a * np.sin(np.deg2rad(spin_phase_angle + phi)) + b


def fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles):
    values, _ = scipy.optimize.curve_fit(sine_fit_function, spin_phase_angles, centers_of_mass, bounds=([0, 0, 0], [np.inf, 360, np.inf]))
    return values


def interpolate_energy(center_of_mass_index, energies):
    return np.exp(np.interp(center_of_mass_index, np.arange(len(energies)), np.log(energies)))


def times_for_sweep(start_time):
    time_step_per_bin = 12 / 72
    return start_time + time_step_per_bin * np.arange(72)


def get_artificial_spin_phase(time):
    arbitrary_offset = 0.4
    rotation_time = 15 * 1_000_000_000
    rotations = arbitrary_offset + time/rotation_time
    fractional, integral = np.modf(rotations)
    return fractional*360


def get_center_of_mass_and_spin_angle_for_single_sweep(sweep_count_rates, spin_angles, energies):
    left, right = get_peak_indices(sweep_count_rates, 4)
    center_of_mass_index = find_peak_center_of_mass_index(left, right, sweep_count_rates)
    energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)

    """
    If the L2 cdf does not provide the spin angle, this will have to be retrieved from SPICE. Otherwise, we can
    interpolate over the spin angles directly from the L2 data
    
    sweep_times = times_for_sweep(sweep_start_time)
    time_of_peak = np.interp(center_of_mass_index, np.arange(len(sweep_times)), sweep_times)
    spin_phase_angle = get_spin_phase_from_spice(time_of_peak)
    """
    interpolated_spin_angle_for_peak = np.interp(center_of_mass_index, np.arange(len(spin_angles)), spin_angles)

    return np.array([energy_at_center_of_mass, interpolated_spin_angle_for_peak])

def extract_course_sweep_energies(data: np.ndarray):
    if data.ndim > 1:
        return data[:, 1:63]
    else:
        return data[1:63]

def read_l2_data(cdf_path: str) -> SwapiL2Data:
    cdf = CDF(cdf_path)
    return SwapiL2Data(cdf.raw_var("epoch")[...],
                       extract_course_sweep_energies(cdf["energy"][...]),
                       extract_course_sweep_energies(cdf["swp_coin_rate"][...]),
                       extract_course_sweep_energies(cdf["spin_angles"][...]))


fig = plt.figure()


def plot_sweeps(data):
    for i in range(len(data.epoch)):
        axes = fig.add_subplot(len(data.epoch)+1, 1, i+1)
        axes.loglog(data.energy, data.coincidence_count_rate[i, :], marker='.', linestyle="None")
        axes.set(xlabel="Energy", ylabel="Count Rate")

def plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass):
    fit_xs = np.arange(0, 360, 3)
    fit_ys = sine_fit_function(fit_xs, a, phi % 360, b)
    plot = fig.add_subplot(len(spin_angles) + 1, 1, len(spin_angles) + 1)

    plot.scatter(spin_angles, centers_of_mass)
    plot.plot(fit_xs, fit_ys)
    plot.set(xlabel="Phase Angle", ylabel="Energy")


def main():
    try:
        file_path = sys.argv[1]
        data = read_l2_data(file_path)
    except:
        print(f"Incorrect file path. Example: python {sys.argv[0]} imap_swapi_l2_sci-1min_20100101_v001_EDITED.cdf")
        exit()

    plot_sweeps(data)

    center_of_mass_and_spin = np.array([get_center_of_mass_and_spin_angle_for_single_sweep(
        data.coincidence_count_rate[i], data.spin_angles[i], data.energy) for i in range(len(data.epoch))])

    centers_of_mass = center_of_mass_and_spin[:, 0]
    spin_angles = center_of_mass_and_spin[:, 1]

    a, phi, b = fit_energy_per_charge_peak_variations(centers_of_mass, spin_angles)

    plot_variation_in_center_of_mass(a, phi, b, spin_angles, centers_of_mass)

    print(f"SW H+ speed: {calculate_sw_speed_h_plus(b)}")
    print(f"SW H+ clock angle: {phi}")
    print(f"A {a}")
    print(f"B {b}")
    print(f"A/B {a/b}")

    fig.legend([f"SW H+ speed: {calculate_sw_speed_h_plus(b):.3f}",
                f"SW H+ clock angle: {phi:.3f}",
               f"A: {a:.3f}",
               f"B: {b:.3f}",
               f"A/B: {(a / b):.5f}"],
               )

    fig.set_figheight(10)
    plt.show()


main()
