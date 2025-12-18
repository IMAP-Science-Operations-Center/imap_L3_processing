from datetime import timedelta

import numpy as np
import scipy
import spiceypy
import uncertainties
from matplotlib import pyplot as plt
from uncertainties import correlated_values, unumpy, umath, wrap
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import PROTON_CHARGE_COULOMBS, PROTON_MASS_KG, METERS_PER_KILOMETER, \
    ONE_SECOND_IN_NANOSECONDS, TT2000_EPOCH
from imap_l3_processing.swapi.l3a.science.speed_calculation import get_peak_indices, find_peak_center_of_mass_index, \
    interpolate_energy, extract_coarse_sweep
from imap_l3_processing.swapi.l3a.utils import DataQualityException, SWAPIDataQualityExceptionType


def sine_fit_function(spin_phase_angle, a, phi, b):
    return a * np.sin(np.deg2rad(phi - spin_phase_angle)) + b


def fit_energy_per_charge_peak_variations(centers_of_mass, spin_phase_angles):
    nominal_centers_of_mass = nominal_values(centers_of_mass)
    min_mass_energy = np.min(nominal_centers_of_mass)
    max_mass_energy = np.max(nominal_centers_of_mass)
    peak_angle = nominal_values(spin_phase_angles[np.argmax(centers_of_mass)])

    initial_parameter_guess = [(max_mass_energy - min_mass_energy) / 2, peak_angle + 90,
                               np.mean(nominal_centers_of_mass)]

    nominal_spin_phase_angles = nominal_values(spin_phase_angles)

    (a, phi, b), pcov = scipy.optimize.curve_fit(
        sine_fit_function, nominal_spin_phase_angles, nominal_centers_of_mass,
        sigma=std_devs(centers_of_mass), bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
        absolute_sigma=True,
        p0=initial_parameter_guess)

    residual = abs(sine_fit_function(np.array(nominal_spin_phase_angles), a, phi, b) - nominal_centers_of_mass)
    reduced_chisq = np.sum(np.square(residual / std_devs(centers_of_mass))) / (len(spin_phase_angles) - 3)

    phi = np.mod(phi, 360)

    if reduced_chisq > 10:
        return False, correlated_values((a, phi, b), pcov), reduced_chisq

    return True, correlated_values((a, phi, b), pcov), reduced_chisq


def get_proton_peak_indices(count_rates):
    return get_peak_indices(count_rates, 4)


def interpolate_angle(center_of_mass_index, spin_angles):
    spin_angle = np.interp(center_of_mass_index, np.arange(len(spin_angles)), np.unwrap(spin_angles, period=360))
    return np.mod(spin_angle, 360)


def calculate_proton_centers_of_mass(coincidence_count_rates, energies, epoch):
    energies_at_center_of_mass = []
    energies_at_center_of_mass_uncertainties = []

    spin_angles_at_center_of_mass = []
    for i in range(len(epoch)):
        rates = coincidence_count_rates[i]
        peak_slice = get_proton_peak_indices(rates)
        center_of_mass_index = find_peak_center_of_mass_index(peak_slice, rates, 13, 4)
        energy_at_center_of_mass = interpolate_energy(center_of_mass_index, energies[i])

        measurement_interval = 1 / 6 * ONE_SECOND_IN_NANOSECONDS
        index_into_full_sweep = (center_of_mass_index + 1)
        time_at_center_of_mass = epoch[i] + measurement_interval * index_into_full_sweep + measurement_interval / 2
        spin_angle = get_angle(time_at_center_of_mass)
        energies_at_center_of_mass.append(energy_at_center_of_mass.nominal_value)
        energies_at_center_of_mass_uncertainties.append(energy_at_center_of_mass.std_dev)
        spin_angles_at_center_of_mass.append(spin_angle)

    return uarray(energies_at_center_of_mass, energies_at_center_of_mass_uncertainties), spin_angles_at_center_of_mass


def get_spin_angle_from_swapi_axis_in_despun_frame(instrument_axis: np.ndarray):
    x, y, _ = instrument_axis
    return np.mod(np.rad2deg(np.atan2(-1 * x, y)), 360)


@wrap
def get_angle(epoch) -> float:
    rotation_matrix = spiceypy.pxform("IMAP_SWAPI", "IMAP_DPS",
                                      spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET"))
    swapi_instrument_axis_in_despun_imap_frame = rotation_matrix @ np.array([0, 0, -1])
    return get_spin_angle_from_swapi_axis_in_despun_frame(swapi_instrument_axis_in_despun_imap_frame)


def calculate_sw_speed(particle_mass, particle_charge, energy):
    if np.size(energy) == 0:
        return np.array([])
    dimensions = np.asanyarray(energy).ndim
    if dimensions > 0:
        if isinstance(np.ravel(energy)[0], uncertainties.UFloat):
            return unumpy.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER
        return np.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER
    else:
        return umath.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER


def calculate_sw_speed_h_plus(energy):
    return calculate_sw_speed(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, energy)


iter2 = 0


def calculate_proton_solar_wind_speed(coincidence_count_rates, energies, epoch, sweep_str):
    global iter2
    energies = extract_coarse_sweep(energies)
    coincidence_count_rates = extract_coarse_sweep(coincidence_count_rates)

    energies_at_center_of_mass, spin_angles_at_center_of_mass = calculate_proton_centers_of_mass(
        coincidence_count_rates, energies, epoch)

    success, (a, phi, b), chisq = fit_energy_per_charge_peak_variations(energies_at_center_of_mass,
                                                                        spin_angles_at_center_of_mass)

    if not success:
        start_of_five_sweeps = TT2000_EPOCH + timedelta(seconds=epoch[0] / 1e9)
        end_of_five_sweeps = TT2000_EPOCH + timedelta(seconds=epoch[-1] / 1e9)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.loglog(energies.T, nominal_values(coincidence_count_rates).T)
        ax2.errorbar(x=nominal_values(spin_angles_at_center_of_mass), xerr=std_devs(spin_angles_at_center_of_mass),
                     y=nominal_values(energies_at_center_of_mass), yerr=std_devs(energies_at_center_of_mass), fmt='o')
        ax2.plot(np.linspace(0, 360, 120), sine_fit_function(np.array(np.linspace(0, 360, 120)), a.n, phi.n, b.n))
        ax2.text(0.8, 0.8, f"reduced_chisq={chisq:.4g}", transform=ax2.transAxes,
                 horizontalalignment='center',
                 verticalalignment='center')
        ax2.text(0.8, 0.7, f"start_time={start_of_five_sweeps}", transform=ax2.transAxes,
                 horizontalalignment='center',
                 verticalalignment='center')
        ax2.text(0.8, 0.6, f"end_time={end_of_five_sweeps}", transform=ax2.transAxes,
                 horizontalalignment='center',
                 verticalalignment='center')
        # plt.title(f"{a=},{phi=},{b=}")
        plt.savefig(f"{sweep_str}.png")
        print(iter2)
        iter2 += 1
        if iter2 > 100:
            exit()
        raise DataQualityException(SWAPIDataQualityExceptionType.ProtonVelocityFit,
                                   f"Failed to fit - chi-squared too large {chisq}")

    proton_sw_speed = calculate_sw_speed_h_plus(b)
    return proton_sw_speed, a, phi, b
