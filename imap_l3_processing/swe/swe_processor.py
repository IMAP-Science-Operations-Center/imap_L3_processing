from datetime import timedelta

import numpy as np
from imap_data_access import upload

from imap_l3_processing import spice_wrapper
from imap_l3_processing.data_utils import find_closest_neighbor
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import calculate_solar_wind_velocity_vector
from imap_l3_processing.swe.l3.models import SweL3Data
from imap_l3_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    filter_and_flatten_regress_parameters, regress, calculate_fit_temperature_density_velocity, rotate_temperature, \
    rotate_dps_vector_to_rtn, halotrunc
from imap_l3_processing.swe.l3.science.pitch_calculations import average_over_look_directions, find_breakpoints, \
    correct_and_rebin, \
    integrate_distribution_to_get_1d_spectrum, integrate_distribution_to_get_inbound_and_outbound_1d_spectrum, \
    calculate_velocity_in_dsp_frame_km_s
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.utils import save_data


class SweProcessor(Processor):
    def process(self):
        dependencies = SweL3Dependencies.fetch_dependencies(self.dependencies)
        self.calculate_moment_products(dependencies)
        output_data = self.calculate_pitch_angle_products(dependencies)
        output_cdf = save_data(output_data)
        upload(output_cdf)

    def calculate_moment_products(self, dependencies: SweL3Dependencies):
        spice_wrapper.furnish()
        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
        config = dependencies.configuration

        spacecraft_potential_history = [config["spacecraft_potential_initial_guess"] for _ in
                                        range(3)]
        halo_core_history = [config["core_halo_breakpoint_initial_guess"] for _ in range(3)]

        core_density_history = [100 for _ in range(3)]
        halo_density_history = [25 for _ in range(3)]

        for i in range(len(swe_epoch)):
            averaged_psd = average_over_look_directions(swe_l2_data.phase_space_density[i],
                                                        np.array(config["geometric_fractions"]),
                                                        config["minimum_phase_space_density_value"])
            spacecraft_potential, halo_core = find_breakpoints(swe_l2_data.energy, averaged_psd,
                                                               spacecraft_potential_history,
                                                               halo_core_history,
                                                               config)

            spacecraft_potential_history = [*spacecraft_potential_history[1:], spacecraft_potential]
            halo_core_history = [*halo_core_history[1:], halo_core]
            corrected_energy_bins = swe_l2_data.energy - spacecraft_potential

            velocity_vectors: np.ndarray = calculate_velocity_in_dsp_frame_km_s(corrected_energy_bins,
                                                                                swe_l2_data.inst_el,
                                                                                swe_l2_data.inst_az_spin_sector[i])

            weights: np.ndarray[float] = compute_maxwellian_weight_factors(dependencies.swe_l1b_data.count_rates[i],
                                                                           dependencies.swe_l2_data.acquisition_duration[
                                                                               i])

            spacecraft_potential_core_breakpoint_index: int = next(
                i for i, energy in enumerate(swe_l2_data.energy) if energy >= spacecraft_potential)
            halo_core_breakpoint_index: int = next(
                i - 1 for i, energy in enumerate(swe_l2_data.energy) if energy > halo_core)

            core_end_index = halo_core_breakpoint_index

            while True:
                filtered_velocity_vectors, filtered_weights, filtered_yreg = filter_and_flatten_regress_parameters(
                    corrected_energy_bins,
                    velocity_vectors,
                    swe_l2_data.phase_space_density[i],
                    weights,
                    spacecraft_potential_core_breakpoint_index, core_end_index)

                fit_function, chisq = regress(filtered_velocity_vectors,
                                              filtered_weights, filtered_yreg)
                core_moments = calculate_fit_temperature_density_velocity(fit_function)

                if 0 < core_moments.density < np.average(core_density_history) * 1.85 or (
                        core_end_index - spacecraft_potential_core_breakpoint_index) <= 3:

                    break
                else:
                    core_end_index -= 1

            core_density_history.append(core_moments.density)
            core_density_history = core_density_history[1:]

            rtn_velocity = rotate_dps_vector_to_rtn(swe_epoch[i],
                                                    np.array(
                                                        [core_moments.velocity_x, core_moments.velocity_y,
                                                         core_moments.velocity_z]))
            rotate_temperature(swe_epoch[i], core_moments.alpha, core_moments.beta)

            halo_end_index = len(swe_l2_data.energy)
            while True:
                filtered_velocity_vectors, filtered_weights, filtered_yreg = filter_and_flatten_regress_parameters(
                    corrected_energy_bins,
                    velocity_vectors,
                    swe_l2_data.phase_space_density[i],
                    weights,
                    halo_core_breakpoint_index, halo_end_index)

                fit_function, chisq = regress(filtered_velocity_vectors,
                                              filtered_weights, filtered_yreg)
                halo_moments = calculate_fit_temperature_density_velocity(fit_function)

                halo_moments.density = halotrunc(halo_moments, halo_core, spacecraft_potential)

                if 0 < halo_moments.density < np.average(halo_density_history) * 1.65 or (
                        halo_end_index - halo_core_breakpoint_index) <= 3:
                    break
                else:
                    halo_end_index -= 1

            halo_density_history.append(halo_moments.density)
            halo_density_history = halo_density_history[1:]

            rtn_velocity = rotate_dps_vector_to_rtn(swe_epoch[i],
                                                    np.array(
                                                        [halo_moments.velocity_x, halo_moments.velocity_y,
                                                         halo_moments.velocity_z]))
            rotate_temperature(swe_epoch[i], halo_moments.alpha, halo_moments.beta)

    def calculate_pitch_angle_products(self, dependencies: SweL3Dependencies) -> SweL3Data:
        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
        swe_epoch_delta = swe_l2_data.epoch_delta
        config = dependencies.configuration
        mag_max_distance = np.timedelta64(int(config['max_mag_offset_in_minutes'] * 60e9), 'ns')
        rebinned_mag_data = find_closest_neighbor(from_epoch=dependencies.mag_l1d_data.epoch,
                                                  from_data=dependencies.mag_l1d_data.mag_data,
                                                  to_epoch=swe_l2_data.acquisition_time,
                                                  maximum_distance=mag_max_distance,
                                                  )

        swapi_l_a_proton_data = dependencies.swapi_l3a_proton_data
        swapi_epoch = swapi_l_a_proton_data.epoch
        solar_wind_vectors = calculate_solar_wind_velocity_vector(swapi_l_a_proton_data.proton_sw_speed,
                                                                  swapi_l_a_proton_data.proton_sw_clock_angle,
                                                                  swapi_l_a_proton_data.proton_sw_deflection_angle)
        swapi_max_distance = timedelta(minutes=config['max_swapi_offset_in_minutes'])
        rebinned_solar_wind_vectors = find_closest_neighbor(from_epoch=swapi_epoch,
                                                            from_data=solar_wind_vectors,
                                                            to_epoch=swe_epoch,
                                                            maximum_distance=swapi_max_distance)

        phase_space_density_by_pitch_angle = []
        energy_spectrum = []
        energy_spectrum_inbound = []
        energy_spectrum_outbound = []
        spacecraft_potential_history = [config["spacecraft_potential_initial_guess"] for _ in
                                        range(3)]
        halo_core_history = [config["core_halo_breakpoint_initial_guess"] for _ in range(3)]

        for i in range(len(swe_epoch)):
            averaged_psd = average_over_look_directions(swe_l2_data.phase_space_density[i],
                                                        np.array(config["geometric_fractions"]),
                                                        config["minimum_phase_space_density_value"])
            spacecraft_potential, halo_core = find_breakpoints(swe_l2_data.energy, averaged_psd,
                                                               spacecraft_potential_history[-3:],
                                                               halo_core_history[-3:],
                                                               config)
            spacecraft_potential_history.append(spacecraft_potential)
            halo_core_history.append(halo_core)

            corrected_energy_bins = swe_l2_data.energy - spacecraft_potential
            missing_mag_data = np.any(np.isnan(rebinned_mag_data[i]))
            if missing_mag_data:
                num_energy_bins = len(config['energy_bins'])
                num_pitch_angle_bins = len(config['pitch_angle_bins'])
                phase_space_density_by_pitch_angle.append(np.full((num_energy_bins, num_pitch_angle_bins), np.nan))
                energy_spectrum.append(np.full(num_energy_bins, np.nan))
                energy_spectrum_inbound.append(np.full(num_energy_bins, np.nan))
                energy_spectrum_outbound.append(np.full(num_energy_bins, np.nan))
            else:

                rebinned_psd = correct_and_rebin(swe_l2_data.phase_space_density[i], corrected_energy_bins,
                                                 swe_l2_data.inst_el,
                                                 swe_l2_data.inst_az_spin_sector[i],
                                                 rebinned_mag_data[i],
                                                 rebinned_solar_wind_vectors[i],
                                                 config, )
                phase_space_density_by_pitch_angle.append(rebinned_psd)
                energy_spectrum.append(integrate_distribution_to_get_1d_spectrum(rebinned_psd, config))
                inbound, outbound = integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(rebinned_psd,
                                                                                                   config)
                energy_spectrum_inbound.append(inbound)
                energy_spectrum_outbound.append(outbound)

        swe_l3_data = SweL3Data(input_metadata=self.input_metadata.to_upstream_data_dependency("sci"),
                                epoch=swe_epoch,
                                epoch_delta=swe_epoch_delta,
                                energy=config["energy_bins"],
                                energy_delta_plus=config["energy_delta_plus"],
                                energy_delta_minus=config["energy_delta_minus"],
                                pitch_angle=config["pitch_angle_bins"],
                                pitch_angle_delta=config["pitch_angle_delta"],
                                phase_space_density_by_pitch_angle=phase_space_density_by_pitch_angle,
                                energy_spectrum=energy_spectrum,
                                energy_spectrum_inbound=energy_spectrum_inbound,
                                energy_spectrum_outbound=energy_spectrum_outbound,
                                spacecraft_potential=spacecraft_potential_history[3:],
                                core_halo_breakpoint=halo_core_history[3:]
                                )
        return swe_l3_data
