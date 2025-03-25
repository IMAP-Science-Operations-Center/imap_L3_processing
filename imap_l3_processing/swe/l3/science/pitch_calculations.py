from __future__ import annotations

from typing import TypeVar

import numpy as np
from scipy.optimize import curve_fit

from imap_l3_processing.constants import ELECTRON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER
from imap_l3_processing.pitch_angles import calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase
from imap_l3_processing.swe.l3.models import SweConfiguration


def piece_wise_model(x: np.ndarray, b0: float, b1: float,
                     b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    return np.log(np.piecewise(x, [x <= b2, (x > b2) & (x <= b4), x > b4],
                               [
                                   lambda x: b0 * np.exp(-b1 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(-b3 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(b4 * (b5 - b3)) * np.exp(-b5 * x),
                               ]))


def find_breakpoints(energies: np.ndarray, averaged_psd: np.ndarray, latest_spacecraft_potentials: list[float],
                     latest_core_halo_break_points: list[float],
                     config: SweConfiguration) -> tuple[
    float, float]:
    log_psd = np.log(averaged_psd)
    slopes = -np.diff(log_psd) / np.diff(energies)
    slope_ratios = slopes[1:] / slopes[:-1]
    numb = np.max(np.nonzero(slope_ratios > config['slope_ratio_cutoff_for_potential_calc']), initial=0)

    energies = energies[:numb]
    log_psd = log_psd[:numb]
    b1: float = slopes[0]
    core_index = np.searchsorted(energies, config["core_energy_for_slope_guess"]) - 1
    halo_index = np.searchsorted(energies, config["halo_energy_for_slope_guess"]) - 1
    b3: float = slopes[core_index]
    b5: float = slopes[halo_index]
    b0: float = np.exp(log_psd[0] + b1 * energies[0])
    initial_guesses = (
        b0, b1, np.average(latest_spacecraft_potentials), b3, np.average(latest_core_halo_break_points), b5)

    first_min_index = 0
    for i in range(1, len(slope_ratios) - 1):
        if slope_ratios[i - 1] > slope_ratios[i] < slope_ratios[i + 1]:
            first_min_index = i
            break

    last_max_index = 0
    for i in reversed(range(1 + first_min_index, len(slopes) - 1)):
        if slopes[i - 1] < slopes[i] > slopes[i + 1]:
            last_max_index = i
            break

    if last_max_index < config["refit_core_halo_breakpoint_index"]:
        delta_b2 = -1.5
        delta_b4 = -10
    else:
        delta_b2 = -1.0
        delta_b4 = 10

    return try_curve_fit_until_valid(energies, log_psd, initial_guesses, latest_spacecraft_potentials[-1],
                                     latest_core_halo_break_points[-1], delta_b2, delta_b4)


def try_curve_fit_until_valid(energies: np.ndarray, log_psd: np.ndarray, initial_guesses: tuple[float, ...],
                              latest_spacecraft_potential: float, latest_core_halo_breakpoint: float,
                              delta_b2: float, delta_b4: float) -> tuple[float, float]:
    b, _ = curve_fit(piece_wise_model, energies, log_psd, initial_guesses)

    def bad_fit(b):
        return (b[1] <= 0 or
                b[3] <= 0 or
                b[5] <= 0 or
                b[2] >= b[4] or
                b[4] <= 15 or
                b[2] <= energies[0] or
                b[2] >= 20 or
                b[2] >= 2 * latest_spacecraft_potential)

    attempt_count = 0

    modified_guesses = list(initial_guesses)
    while bad_fit(b):
        if attempt_count < 3:
            modified_guesses[2] += delta_b2
            modified_guesses[4] += delta_b4
            b, _ = curve_fit(piece_wise_model, energies, log_psd, tuple(modified_guesses))
            attempt_count += 1
        else:
            return latest_spacecraft_potential, latest_core_halo_breakpoint
    return b[2], b[4]


def average_over_look_directions(phase_space_density: np.ndarray, geometric_weights: np.ndarray,
                                 minimum_psd_value: float) -> np.ndarray:
    enforced_minimum_psd = np.maximum(phase_space_density, minimum_psd_value)
    weighted_average = np.sum(enforced_minimum_psd * geometric_weights, axis=-1) / np.sum(geometric_weights)
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


def rebin_by_pitch_angle(flux, pitch_angles, energies, config: SweConfiguration) -> np.ndarray:
    all_the_bins = rebin_by_pitch_angle_and_gyrophase(flux, pitch_angles, None, energies, config)
    remove_gyrophase_axis = all_the_bins[:, :, 0]
    return remove_gyrophase_axis


def _rebin(flux, pitch_angles, energies, config: SweConfiguration, gyrophase: np.ndarray = None) -> np.ndarray:
    pitch_angle_bins = np.array(config["pitch_angle_bins"])
    pitch_angle_delta = np.array(config["pitch_angle_deltas"])
    energy_bins = config["energy_bins"]
    mask_psd = flux > 0
    psd_greater_than_zero = flux[mask_psd]
    pitch_angles_for_masked_psd = pitch_angles[mask_psd]
    energies_for_masked_psd = energies[mask_psd]
    pitch_angle_left_edges = pitch_angle_bins - pitch_angle_delta
    pitch_angle_right_edges = pitch_angle_bins + pitch_angle_delta
    num_pitch_bins = len(pitch_angle_bins)
    num_energy_bins = len(energy_bins)

    if gyrophase is not None:
        gyrophase_bins = np.array(config["gyrophase_bins"])
        gyrophase_delta = np.array(config["gyrophase_deltas"])
        gyrophase_left_edges = gyrophase_bins - gyrophase_delta
        gyrophase_right_edges = gyrophase_bins + gyrophase_delta
        gyrophase_for_masked_psd = gyrophase[mask_psd]
        num_gyrophase_bins = len(gyrophase_bins)
        gyrophase_bins_range = range(len(gyrophase_bins))
    else:
        gyrophase_bins_range = [0]
        num_gyrophase_bins = 1

    rebinned = np.full((num_energy_bins, num_pitch_bins, num_gyrophase_bins), np.nan, dtype=float)
    for g in gyrophase_bins_range:
        for j in range(num_pitch_bins):
            mask_pitch_angle = (pitch_angles_for_masked_psd >= pitch_angle_left_edges[j]) & (
                    pitch_angles_for_masked_psd < pitch_angle_right_edges[j])

            if gyrophase is not None:
                mask_gyrophase = (gyrophase_for_masked_psd >= gyrophase_left_edges[g]) & (
                        gyrophase_for_masked_psd < gyrophase_right_edges[g])
            else:
                mask_gyrophase = np.ones_like(mask_pitch_angle, dtype=bool)

            psd_by_pitch_angle = psd_greater_than_zero[mask_pitch_angle & mask_gyrophase]
            energy_by_pitch_angle = energies_for_masked_psd[mask_pitch_angle & mask_gyrophase]

            # maintain in new method
            if len(energy_by_pitch_angle) < 2:
                continue
                # this requires some thinking to not end up with all fill val
            sorted_energies = np.sort(energy_by_pitch_angle)
            overall_max_energy_for_pitch_angle = sorted_energies[-1]
            overall_second_min_energy_for_pitch_angle = sorted_energies[1]

            for i, center in enumerate(energy_bins):
                left = config["energy_bin_low_multiplier"] * center
                right = config["energy_bin_high_multiplier"] * center

                mask_energy = (energy_by_pitch_angle > left) & (energy_by_pitch_angle < right)

                energy_to_fit = energy_by_pitch_angle[mask_energy]
                psd_to_fit = psd_by_pitch_angle[mask_energy]
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
                    log_psd_to_fit = np.log(psd_to_fit)
                    intercept, slope = np.polynomial.polynomial.polyfit(log_energy_to_fit, log_psd_to_fit, 1)
                    log_psd_to_nom = slope * np.log(center) + intercept
                    value = np.exp(log_psd_to_nom)

                    rebinned[i, j, g] = value

    return rebinned


def rebin_by_pitch_angle_and_gyrophase(psd, pitch_angles, gyrophase, energies, config: SweConfiguration) -> np.ndarray:
    return _rebin(psd, pitch_angles, energies, config, gyrophase=gyrophase)


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
GYROPHASE_BINS = TypeVar("GYROPHASE_BINS")


def correct_and_rebin(flux_or_psd: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS)],
                      solar_wind_vector: np.ndarray[(3,)],
                      dsp_velocities: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS, 3,)],
                      mag_vector: np.ndarray[(E_BINS, SPIN_SECTORS, 3,)],
                      config: SweConfiguration) -> tuple[np.ndarray, np.ndarray]:
    velocity_in_sw_frame = calculate_velocity_in_sw_frame(dsp_velocities, solar_wind_vector)
    energy_in_sw_frame = calculate_energy_in_ev_from_velocity_in_km_per_second(velocity_in_sw_frame)
    pitch_angle = calculate_pitch_angle(velocity_in_sw_frame, mag_vector[..., np.newaxis, :])
    gyrophase = calculate_gyrophase(velocity_in_sw_frame, mag_vector[..., np.newaxis, :])

    rebinned_by_pa = rebin_by_pitch_angle(flux_or_psd, pitch_angle, energy_in_sw_frame, config)
    rebinned_by_pa_and_gyro = rebin_by_pitch_angle_and_gyrophase(flux_or_psd, pitch_angle, gyrophase,
                                                                 energy_in_sw_frame, config)

    return rebinned_by_pa, rebinned_by_pa_and_gyro


def integrate_distribution_to_get_1d_spectrum(psd_by_energy_and_pitch_angle: np.ndarray[(E_BINS, PITCH_ANGLE_BINS)],
                                              config: SweConfiguration) -> np.ndarray[E_BINS]:
    pitch_angle_bin_factors = np.sin(np.deg2rad(config["pitch_angle_bins"])) * 2 * np.deg2rad(
        config["pitch_angle_deltas"]) / 2

    return np.nansum(psd_by_energy_and_pitch_angle * pitch_angle_bin_factors, axis=1)


def integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(
        psd_by_pitch_angle: np.ndarray[(E_BINS, PITCH_ANGLE_BINS)],
        config: SweConfiguration) -> (np.ndarray[E_BINS], np.ndarray[E_BINS]):
    pitch_angle_bin_factors = np.sin(np.deg2rad(config["pitch_angle_bins"])) * 2 * np.deg2rad(
        config["pitch_angle_deltas"])
    pitch_less_than_90 = np.array(config["pitch_angle_bins"]) < 90

    spectrum_a = np.nansum((psd_by_pitch_angle * pitch_angle_bin_factors)[:, pitch_less_than_90], axis=1)
    spectrum_b = np.nansum((psd_by_pitch_angle * pitch_angle_bin_factors)[:, ~pitch_less_than_90], axis=1)

    if spectrum_a[config["in_vs_out_energy_index"]] > spectrum_b[config["in_vs_out_energy_index"]]:
        inbound = spectrum_b
        outbound = spectrum_a
    else:
        inbound = spectrum_a
        outbound = spectrum_b

    return inbound, outbound


def swe_rebin_intensity_by_pitch_angle_and_gyrophase(intensity_data: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS)],
                                                     counts: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS)],
                                                     dsp_velocities: np.ndarray[(E_BINS, SPIN_SECTORS, CEMS, 3)],
                                                     mag_vectors: np.ndarray[([(E_BINS, SPIN_SECTORS, 3,)])],
                                                     config: SweConfiguration) -> [np.ndarray]:
    normalized_velocities = calculate_unit_vector(dsp_velocities)
    normalized_mag_vectors = calculate_unit_vector(mag_vectors)

    pitch_angles = calculate_pitch_angle(normalized_velocities, normalized_mag_vectors[..., np.newaxis, :])
    gyrophases = calculate_gyrophase(normalized_velocities, normalized_mag_vectors[..., np.newaxis, :])

    num_pitch_angle_bins = len(config["pitch_angle_bins"])
    num_gyrophase_bins = len(config["gyrophase_bins"])
    output_shape_pa_and_gyro = (intensity_data.shape[0], num_pitch_angle_bins, num_gyrophase_bins)
    output_shape_pa_only = (intensity_data.shape[0], num_pitch_angle_bins)

    rebinned_summed_by_pa_and_gyro = np.zeros(shape=output_shape_pa_and_gyro)
    rebinned_summed_pa_only = np.zeros(shape=output_shape_pa_only)

    rebinned_count_by_pa_and_gyro = np.zeros(shape=output_shape_pa_and_gyro)
    rebinned_count_pa_only = np.zeros(shape=output_shape_pa_only)

    rebinned_summed_counts_by_pa_and_gyro = np.zeros(shape=output_shape_pa_and_gyro)
    rebinned_summed_counts_by_pa_only = np.zeros(shape=output_shape_pa_only)

    for i in range(intensity_data.shape[0]):
        for pitch_angle, gyrophase, intensity, count in zip(
                np.ravel(pitch_angles[i]),
                np.ravel(gyrophases[i]),
                np.ravel(intensity_data[i]),
                np.ravel(counts[i])):
            if not (np.isnan(intensity) or np.isnan(pitch_angle)):
                pitch_angle_bin = next((i for i, (center, delta) in
                                        enumerate(zip(config["pitch_angle_bins"], config["pitch_angle_deltas"])) if
                                        center - delta <= pitch_angle < center + delta),
                                       num_pitch_angle_bins - 1)
                if not np.isnan(gyrophase):
                    gyrophase_bin = next((i for i, (center, delta) in
                                          enumerate(zip(config["gyrophase_bins"], config["gyrophase_deltas"])) if
                                          center - delta <= gyrophase < center + delta),
                                         num_gyrophase_bins - 1)

                    rebinned_summed_by_pa_and_gyro[i, pitch_angle_bin, gyrophase_bin] += intensity
                    rebinned_count_by_pa_and_gyro[i, pitch_angle_bin, gyrophase_bin] += 1

                    rebinned_summed_counts_by_pa_and_gyro[i, pitch_angle_bin, gyrophase_bin] += count

                rebinned_summed_pa_only[i, pitch_angle_bin] += intensity
                rebinned_count_pa_only[i, pitch_angle_bin] += 1
                rebinned_summed_counts_by_pa_only[i, pitch_angle_bin] += count

    rebinned_summed_counts_by_pa_and_gyro[rebinned_summed_counts_by_pa_and_gyro == 0] = np.nan
    rebinned_summed_counts_by_pa_only[rebinned_summed_counts_by_pa_only == 0] = np.nan

    averaged_rebinned_intensity_by_pa_and_gyro = np.divide(
        rebinned_summed_by_pa_and_gyro,
        rebinned_count_by_pa_and_gyro,
        out=np.full_like(
            rebinned_count_by_pa_and_gyro, np.nan),
        where=rebinned_count_by_pa_and_gyro != 0)

    averaged_rebinned_intensity_by_pa_only = np.divide(rebinned_summed_pa_only,
                                                       rebinned_count_pa_only,
                                                       out=np.full_like(
                                                           rebinned_count_pa_only,
                                                           np.nan),
                                                       where=rebinned_count_pa_only != 0)

    uncertainties_by_pa_and_gyro = averaged_rebinned_intensity_by_pa_and_gyro / np.sqrt(
        rebinned_summed_counts_by_pa_and_gyro)
    uncertainties_by_pa_only = averaged_rebinned_intensity_by_pa_only / np.sqrt(rebinned_summed_counts_by_pa_only)

    return (
        averaged_rebinned_intensity_by_pa_and_gyro, averaged_rebinned_intensity_by_pa_only,
        uncertainties_by_pa_and_gyro,
        uncertainties_by_pa_only)
