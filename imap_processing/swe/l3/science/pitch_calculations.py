from __future__ import annotations

from typing import TypeVar

import numpy as np
from scipy.optimize import curve_fit

from imap_processing.constants import ELECTRON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER
from imap_processing.pitch_angles import calculate_pitch_angle
from imap_processing.swe.l3.models import SweConfiguration


def piece_wise_model(x: np.ndarray, b0: float, b1: float,
                     b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    return np.log(np.piecewise(x, [x <= b2, (x > b2) & (x <= b4), x > b4],
                               [
                                   lambda x: b0 * np.exp(-b1 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(-b3 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(b4 * (b5 - b3)) * np.exp(-b5 * x),
                               ]))


def find_breakpoints(energies: np.ndarray, flux: np.ndarray, initial_spacecraft_potential_guess: float,
                     initial_core_halo_break_point_guess: float) -> tuple[
    np.ndarray, np.ndarray]:
    log_flux = np.log(flux)
    slope = -np.diff(log_flux) / np.diff(energies)
    xsratio = slope[1:] / slope[:-1]
    numb = np.max(np.nonzero(xsratio > 0.55), initial=0)

    energies = energies[:numb]
    log_flux = log_flux[:numb]

    b5 = (log_flux[10] - log_flux[11]) / (energies[11] - energies[10])
    b3 = (log_flux[5] - log_flux[6]) / (energies[6] - energies[5])
    b1 = (log_flux[0] - log_flux[1]) / (energies[1] - energies[0])
    b0 = np.exp(log_flux[0] + b1 * energies[0])
    initial_guesses = (b0, b1, initial_spacecraft_potential_guess, b3, initial_core_halo_break_point_guess, b5)

    fit, covariance = curve_fit(piece_wise_model, energies, log_flux, initial_guesses)

    return fit[2], fit[4]


def average_flux(flux_data: np.ndarray, geometric_weights: np.ndarray) -> np.ndarray:
    weighted_average = np.sum(flux_data * geometric_weights, axis=-1) / np.sum(geometric_weights)
    return np.mean(weighted_average, axis=1)


def calculate_look_directions(inst_el: np.ndarray, inst_az: np.ndarray) -> np.ndarray:
    inst_az_rad = np.deg2rad(inst_az)[..., np.newaxis]
    inst_el_rad = np.deg2rad(inst_el)
    z = np.sin(inst_el_rad)
    cos_el = np.cos(inst_el_rad)
    x = - cos_el * np.sin(inst_az_rad)
    y = cos_el * np.cos(inst_az_rad)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def calculate_velocity_in_dsp_frame_km_s(energy: np.ndarray, inst_el: np.ndarray, inst_az: np.ndarray) -> np.ndarray:
    particle_direction = - calculate_look_directions(inst_el, inst_az)
    speed_meters_per_second = np.sqrt(energy * PROTON_CHARGE_COULOMBS * 2 / ELECTRON_MASS_KG)
    speed_km_per_second = speed_meters_per_second / METERS_PER_KILOMETER

    for i in range(len(particle_direction)):
        particle_direction[i] *= speed_km_per_second[i]

    return particle_direction


def rebin_by_pitch_angle(flux, pitch_angles, energies, config: SweConfiguration) -> np.ndarray[
    (E_BINS, PITCH_ANGLE_BINS)]:
    pitch_angle_bins = np.array(config["pitch_angle_bins"])
    pitch_angle_delta = np.array(config["pitch_angle_delta"])
    energy_bins = config["energy_bins"]
    pitch_angle_left_edges = pitch_angle_bins - pitch_angle_delta
    pitch_angle_right_edges = pitch_angle_bins + pitch_angle_delta
    mask_flux = flux > 0
    flux_greater_than_zero = flux[mask_flux]
    pitch_angles_for_masked_flux = pitch_angles[mask_flux]
    energies_for_masked_flux = energies[mask_flux]

    num_pitch_bins = len(pitch_angle_bins)
    num_energy_bins = len(energy_bins)

    rebinned = np.zeros((num_energy_bins, num_pitch_bins), dtype=float)

    for j in range(num_pitch_bins):
        mask_pitch_angle = (pitch_angles_for_masked_flux >= pitch_angle_left_edges[j]) & (
                pitch_angles_for_masked_flux < pitch_angle_right_edges[j])

        flux_by_pitch_angle = flux_greater_than_zero[mask_pitch_angle]
        energy_by_pitch_angle = energies_for_masked_flux[mask_pitch_angle]
        if len(energy_by_pitch_angle) < 2:
            continue
        sorted_energies = np.sort(energy_by_pitch_angle)
        overall_max_energy_for_pitch_angle = sorted_energies[-1]
        overall_second_min_energy_for_pitch_angle = sorted_energies[1]

        for i, center in enumerate(energy_bins):
            left = config["energy_bin_low_multiplier"] * center
            right = config["energy_bin_high_multiplier"] * center

            mask_energy = (energy_by_pitch_angle > left) & (energy_by_pitch_angle < right)

            energy_to_fit = energy_by_pitch_angle[mask_energy]
            flux_to_fit = flux_by_pitch_angle[mask_energy]
            if len(energy_to_fit) < 2:
                continue

            closest_energy_ratio = np.min(np.abs(energy_to_fit / center - 1))
            max_within_window = np.max(energy_to_fit)
            min_within_window = np.min(energy_to_fit)

            has_points_on_both_sides = max_within_window > center and min_within_window < center
            is_overall_max_within_window = overall_max_energy_for_pitch_angle in energy_to_fit
            max_within_case = is_overall_max_within_window and closest_energy_ratio < config[
                "high_energy_proximity_threshold"]

            is_overall_second_min_within_window = overall_second_min_energy_for_pitch_angle in energy_to_fit

            mins_within_case = is_overall_second_min_within_window and (closest_energy_ratio < config[
                "low_energy_proximity_threshold"] or has_points_on_both_sides)
            max_and_mins_outside_window_case = (not is_overall_max_within_window
                                                and not is_overall_second_min_within_window
                                                and has_points_on_both_sides)

            if max_within_case or mins_within_case or max_and_mins_outside_window_case:
                log_energy_to_fit = np.log(energy_to_fit)
                log_flux_to_fit = np.log(flux_to_fit)
                intercept, slope = np.polynomial.polynomial.polyfit(log_energy_to_fit, log_flux_to_fit, 1)
                log_flux_to_nom = slope * np.log(center) + intercept
                rebinned[i, j] = np.exp(log_flux_to_nom)

    return rebinned


def calculate_velocity_in_sw_frame(velocity_in_despun_frame: np.ndarray[(..., 3)],
                                   solar_wind_velocity: np.ndarray[(3,)]) -> np.ndarray[(..., 3)]:
    return velocity_in_despun_frame - solar_wind_velocity


def calculate_energy_in_ev_from_velocity_in_km_per_second(velocity: np.ndarray[(..., 3)]):
    joule_to_ev = 1 / PROTON_CHARGE_COULOMBS

    velocity_m_s = velocity * 1e3

    speed_squared = np.sum(velocity_m_s ** 2, axis=-1)
    energy_joules = 0.5 * ELECTRON_MASS_KG * speed_squared

    return energy_joules * joule_to_ev


E_BINS = TypeVar("E_BINS")
SPIN_SECTORS = TypeVar("SPIN_SECTORS")
CEMS = TypeVar("CEMS")
PITCH_ANGLE_BINS = TypeVar("PITCH_ANGLE_BINS")


def correct_and_rebin(flux_or_psd: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS)],
                      energy_bins_minus_potential: np.ndarray[(E_BINS,)],
                      inst_el: np.ndarray[(CEMS,)],
                      inst_az: np.ndarray[E_BINS, SPIN_SECTORS, CEMS],
                      mag_vector: np.ndarray[(E_BINS, SPIN_SECTORS, 3,)],
                      solar_wind_vector: np.ndarray[(3,)],
                      config: SweConfiguration) -> np.ndarray[(E_BINS, PITCH_ANGLE_BINS)]:
    despun_velocity = calculate_velocity_in_dsp_frame_km_s(energy_bins_minus_potential, inst_el, inst_az)
    velocity_in_sw_frame = calculate_velocity_in_sw_frame(despun_velocity, solar_wind_vector)
    pitch_angle = calculate_pitch_angle(velocity_in_sw_frame, mag_vector[..., np.newaxis, :])
    energy_in_sw_frame = calculate_energy_in_ev_from_velocity_in_km_per_second(velocity_in_sw_frame)

    return rebin_by_pitch_angle(flux_or_psd, pitch_angle, energy_in_sw_frame, config)


def integrate_distribution_to_get_1d_spectrum(psd_by_pitch_angle: np.ndarray[(E_BINS, PITCH_ANGLE_BINS)],
                                              config: SweConfiguration) -> np.ndarray[E_BINS]:
    pitch_angle_bin_factors = np.sin(np.deg2rad(config["pitch_angle_bins"])) * 2 * np.deg2rad(
        config["pitch_angle_delta"]) / 2

    return np.sum(psd_by_pitch_angle * pitch_angle_bin_factors, axis=1)


def integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(
        psd_by_pitch_angle: np.ndarray[(E_BINS, PITCH_ANGLE_BINS)],
        config: SweConfiguration) -> (np.ndarray[E_BINS], np.ndarray[E_BINS]):
    pitch_angle_bin_factors = np.sin(np.deg2rad(config["pitch_angle_bins"])) * 2 * np.deg2rad(
        config["pitch_angle_delta"])
    pitch_less_than_90 = np.array(config["pitch_angle_bins"]) < 90

    spectrum_a = np.sum((psd_by_pitch_angle * pitch_angle_bin_factors)[:, pitch_less_than_90], axis=1)
    spectrum_b = np.sum((psd_by_pitch_angle * pitch_angle_bin_factors)[:, ~pitch_less_than_90], axis=1)

    if spectrum_a[config["in_vs_out_energy_index"]] > spectrum_b[config["in_vs_out_energy_index"]]:
        inbound = spectrum_b
        outbound = spectrum_a
    else:
        inbound = spectrum_a
        outbound = spectrum_b

    return inbound, outbound
