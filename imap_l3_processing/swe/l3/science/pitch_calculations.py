from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipy

from imap_l3_processing.constants import ELECTRON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER
from imap_l3_processing.pitch_angles import calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase
from imap_l3_processing.swe.l3.models import SweConfiguration
from imap_l3_processing.swe.quality_flags import SweL3Flags

def mec_breakpoint_finder(energies: np.ndarray, averaged_psd: np.ndarray) -> tuple[float, float, SweL3Flags]:
    """
    Input:
        energies - energy bins
        averaged_psd - phase space density, either averaged over all CEMs or individual CEMs
    Output:
        sc_pot_output, ch_break_output, total_flag - tuple for spacecraft potential, core-halo break point, and all flags thrown
    """
    log_energy = np.log(energies)
    log_psd = np.log(averaged_psd)

    # Check to see if first 4 points are nearly linear
    # If True, then potential is likely less than 2.7 V (lower than first energy bin)
    def line_model(params, x):
        return params[0] + params[1] * x
    from scipy import odr
    odr_model = odr.Model(line_model)
    x = log_energy[:4]
    y = log_psd[:4]
    mydata = odr.RealData(x=x,y=y)
    myodr = odr.ODR(mydata, odr_model, beta0=[y.max(),-5])
    myodr.set_job(fit_type=2)
    myoutput= myodr.run()
    if myoutput.res_var <= 0.01:
        FALLBACK_POTENTIAL_ESTIMATE = SweL3Flags.FALLBACK_POTENTIAL_ESTIMATE
        return_value = 2.5
    else:
        FALLBACK_POTENTIAL_ESTIMATE = SweL3Flags.NONE
    
    # Use a smoothed spline on log_psd for spectral break finding routine as a fall back only!
    # Mirror real point as fake point to left of first energy bin to improve spline concavity
    ewidth = np.nanmean(log_energy[1:] - log_energy[:-1])
    from scipy.interpolate import UnivariateSpline as uspline
    spline = uspline(np.concatenate([[log_energy[0]-ewidth],log_energy]), 
                     np.concatenate([[log_psd[2]],log_psd]), s=.25)
    spline_energies = np.geomspace(energies.min()*np.exp(-ewidth), energies.max(), 100)
    spline_derivative = spline.derivative(2)
    curvature = spline_derivative(np.log(spline_energies))
    try:
        peaks = scipy.signal.find_peaks(curvature)[0]
        sc_pot = spline_energies[peaks[0]]
        ch_break = spline_energies[peaks[1]]
        BACKUP_SPLINE_UNRESOLVED = SweL3Flags.NONE
    except:
        # Spline peak finder did not work
        BACKUP_SPLINE_UNRESOLVED = SweL3Flags.BACKUP_SPLINE_UNRESOLVED
        sc_pot = np.nan
        ch_break = np.nan

    def piece_wise_model_mec(x, b0, b1, b2, b3):
        """
        Modified Piecewise to fit Potential and Core-Halo Break separately
        The breakpoint is b2
        """
        return np.piecewise(x, [x<=b2, x>b2], 
                               [lambda x: b0 - b1*x, lambda x: b0 + b2*(b3-b1) - b3*x])

    def refine_breakpoint_value(energy, psd, breakpoint_value, num_points):
        """
        Function to use lines from num_points to left and right to find intersection
        energy at which the lines intersect is the refined_breakpoint
        """
        # Find Nearest energy bin to breakpoint_value
        nearest_energy_idx = np.argmin(np.abs(energy-breakpoint_value))
        if np.abs(breakpoint_value - energy[nearest_energy_idx])/energy[nearest_energy_idx] <= .075:
            # Breakpoint_value was within FWHM of nearest energy bin
            # return that energy bin as refined_breakpoint
            return energy[nearest_energy_idx]
        # Get left and right spectrum of breakpoint_value
        e_left = energy[energy < breakpoint_value]
        e_right = energy[energy > breakpoint_value]
        psd_left = psd[energy < breakpoint_value]
        psd_right = psd[energy > breakpoint_value]
        if len(e_left) < num_points:
            # Not enough points to the left (really only possible for s/c potential)
            return breakpoint_value
        # Fit a line to the num_points left and right of breakpoint_value
        z_left = np.polyfit(e_left[-num_points:], psd_left[-num_points:], 1)
        z_right = np.polyfit(e_right[:num_points], psd_right[:num_points], 1)
        # Determine energy of their intersection
        refined_breakpoint = (z_left[1]-z_right[1]) / (z_right[0]-z_left[0])
        if (refined_breakpoint < z_left[-1]) | (refined_breakpoint > z_right[0]):
            # Refined breakpoint lies outside of expected range
            return breakpoint_value
        return refined_breakpoint
    
    # Prepare masking for the two separate fits
    mask_sc = log_energy <= np.log(30)
    mask_ch = (log_energy > np.log(30)) & (log_energy < np.log(400))

    log_energy_sc = log_energy[mask_sc]; log_psd_sc = log_psd[mask_sc]
    log_energy_ch = log_energy[mask_ch]; log_psd_ch = log_psd[mask_ch]

    fitting_model = piece_wise_model_mec
    # Try Spacecraft Potential Fit
    POTENTIAL_FIT_UNCONVERGED = SweL3Flags.NONE
    if FALLBACK_POTENTIAL_ESTIMATE == SweL3Flags.NONE:
        try:
            initial_guess = [log_psd[0],1,7,1]
            z, cov = scipy.optimize.curve_fit(fitting_model, np.exp(log_energy_sc), log_psd_sc, p0=initial_guess)
            # Make sure the fit converged
            if ((z[0] == initial_guess[0]) | (z[1] == initial_guess[1]) 
                | (z[2] == initial_guess[2]) | (z[3] == initial_guess[3])):
                # Fall back on Spline method
                # Fit did not converge
                POTENTIAL_FIT_UNCONVERGED = SweL3Flags.POTENTIAL_FIT_UNCONVERGED
                sc_pot_output = sc_pot
            else:
                # Fit worked
                # Check whether breakpoint has two points to left and right
                # If so, then find intersection of linear fits to each side
                sc_pot_output = refine_breakpoint_value(np.exp(log_energy_sc), log_psd_sc, z[2], 2)
                # spline not used
                BACKUP_SPLINE_UNRESOLVED = SweL3Flags.NONE
        except:
            # Fall back on Spline method
            # Fit did not converge
            BACKUP_SPLINE_UNRESOLVED = SweL3Flags.BACKUP_SPLINE_UNRESOLVED
            sc_pot_output = sc_pot
    else:
        sc_pot_output = return_value
    # Try Core-Halo Breakpoint Fit
    BREAKPOINT_FIT_UNCONVERGED = SweL3Flags.NONE
    try:
        initial_guess = [log_psd_ch[0],1,65,1]
        z, cov = scipy.optimize.curve_fit(fitting_model, np.exp(log_energy_ch), log_psd_ch, p0=initial_guess)
        # Make sure the fit converged
        if ((z[0] == initial_guess[0]) & (z[1] == initial_guess[1]) 
            & (z[2] == initial_guess[2]) & (z[3] == initial_guess[3])):
            # Fall back on Spline method
            # Fit did not converge
            BREAKPOINT_FIT_UNCONVERGED = SweL3Flags.BREAKPOINT_FIT_UNCONVERGED
            ch_break_output = ch_break
        else:
            # Fit worked
            # Check whether breakpoint has three points to left and right
            # If so, then find intersection of linear fits to each side
            ch_break_output = refine_breakpoint_value(np.exp(log_energy_ch), log_psd_ch, z[2], 3)
    except:
        # Fall back on Spline method
        # Fit did not converge
        BREAKPOINT_FIT_UNCONVERGED = SweL3Flags.BREAKPOINT_FIT_UNCONVERGED
        ch_break_output = ch_break
    return sc_pot_output, ch_break_output, FALLBACK_POTENTIAL_ESTIMATE | BACKUP_SPLINE_UNRESOLVED | POTENTIAL_FIT_UNCONVERGED | BREAKPOINT_FIT_UNCONVERGED


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
    x = cos_el * np.cos(inst_az_rad)
    y = cos_el * np.sin(inst_az_rad)
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
