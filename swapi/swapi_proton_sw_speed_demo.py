import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

from fake_data_generator import get_sweep_voltages, get_k_factor, get_spin_phase_using_spice, generate_sweeps


def times_for_sweep(start_time):
    # times are reversed because the sweep measures from highest to lowest energy
    return start_time + 62/6 - np.arange(62)/6


def get_peak_indices(count_rates, width):
    left_min_index = np.min(np.argmax(count_rates))
    right_min_index = np.max(np.argmax(count_rates))
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


def calculate_sw_speed_h_plus(energy):
    proton_charge = 1.602176634e-19
    proton_mass = 1.67262192595e-27
    return np.sqrt(2 * energy * proton_charge / proton_mass)


def interpolate_energy(center_of_mass_index, energies):
    return np.exp(np.interp(center_of_mass_index, np.arange(len(energies)), np.log(energies)))


def get_center_of_mass_and_spin_angle_for_single_sweep(sweep_count_rates, sweep_start_time):
    voltages = get_sweep_voltages(sweep_table_id=0)
    left, right = get_peak_indices(sweep_count_rates, 4)
    energies = voltages * get_k_factor()
    center_of_mass_index = find_peak_center_of_mass_index(left, right, sweep_count_rates)
    energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)
    sweep_times = times_for_sweep(sweep_start_time)
    time_of_peak = np.interp(center_of_mass_index, np.arange(len(sweep_times)), sweep_times)
    spin_phase_angle = get_spin_phase_using_spice(time_of_peak)

    return energy_at_center_of_mass, spin_phase_angle


def plot_swapi_sweeps(sweeps):
    voltages = get_sweep_voltages()
    energies = voltages * get_k_factor()
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].loglog(energies, sweeps[0][1], marker=".")
    axes[0, 1].loglog(energies, sweeps[1][1], marker=".")
    axes[0, 2].loglog(energies, sweeps[2][1], marker=".")
    axes[1, 0].loglog(energies, sweeps[3][1], marker=".")
    axes[1, 1].loglog(energies, sweeps[4][1], marker=".")


def run():
    sweeps = generate_sweeps()
    plot_swapi_sweeps(sweeps)

    centers_of_mass = []
    spin_phase_angles = []
    for start_time, sweep in sweeps:
        center_of_mass, spin_phase_angle = get_center_of_mass_and_spin_angle_for_single_sweep(sweep, start_time)
        centers_of_mass.append(center_of_mass)
        spin_phase_angles.append(spin_phase_angle)

    a, phi, b = fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles)

    print(f"SW H+ speed: {calculate_sw_speed_h_plus(b)}")
    print(f"SW H+ clock angle: {phi}")

    plt.scatter(spin_phase_angles, centers_of_mass)

    plt.show()


if __name__ == '__main__':
    run()