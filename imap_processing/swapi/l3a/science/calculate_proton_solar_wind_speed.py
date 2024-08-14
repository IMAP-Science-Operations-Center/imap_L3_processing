import numpy as np
import scipy


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
    values, _ = scipy.optimize.curve_fit(sine_fit_function, spin_phase_angles, centers_of_mass,
                                         bounds=([0, 0, 0], [np.inf, 360, np.inf]))
    return values


def interpolate_energy(center_of_mass_index, energies):
    return np.exp(np.interp(center_of_mass_index, np.arange(len(energies)), np.log(energies)))


def times_for_sweep(start_time):
    time_step_per_bin = 12 / 72
    return start_time + time_step_per_bin * np.arange(72)


def get_artificial_spin_phase(time):
    arbitrary_offset = 0.4
    rotation_time = 15 * 1_000_000_000
    rotations = arbitrary_offset + time / rotation_time
    fractional, integral = np.modf(rotations)
    return fractional * 360


def extract_coarse_sweep(data: np.ndarray):
    if data.ndim > 1:
        return data[:, 1:63]
    else:
        return data[1:63]


def calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch):
    energies_at_center_of_mass = []
    spin_angles_at_center_of_mass = []
    for i in range(len(epoch)):
        rates = coincidence_count_rates[i]
        angles = spin_angles[i]
        left, right = get_peak_indices(rates, 4)
        center_of_mass_index = find_peak_center_of_mass_index(left, right, rates)
        energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies)
        """
            If the L2 cdf does not provide the spin angle, this will have to be retrieved from SPICE. Otherwise, we can
            interpolate over the spin angles directly from the L2 data
        
            sweep_times = times_for_sweep(sweep_start_time)
            time_of_peak = np.interp(center_of_mass_index, np.arange(len(sweep_times)), sweep_times)
            spin_phase_angle = get_spin_phase_from_spice(time_of_peak)
            """
        interpolated_spin_angle_for_peak = np.interp(center_of_mass_index, np.arange(len(angles)), angles)
        result = energy_at_center_of_mass, interpolated_spin_angle_for_peak
        energy, spin_angle = result
        energies_at_center_of_mass.append(energy)
        spin_angles_at_center_of_mass.append(spin_angle)

    return energies_at_center_of_mass, spin_angles_at_center_of_mass


def calculate_proton_solar_wind_speed(coincidence_count_rates, spin_angles, energies, epoch):
    epoch = extract_coarse_sweep(epoch)
    energies = extract_coarse_sweep(energies)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)
    spin_angles = extract_coarse_sweep(spin_angles)

    energies_at_center_of_mass, spin_angles_at_center_of_mass = calculate_proton_centers_of_mass(coincidence_count_rates, spin_angles, energies, epoch)

    a, phi, b = fit_energy_per_charge_peak_variations(energies_at_center_of_mass, spin_angles_at_center_of_mass)

    proton_sw_speed = calculate_sw_speed_h_plus(b)
    return proton_sw_speed, a, phi, b

