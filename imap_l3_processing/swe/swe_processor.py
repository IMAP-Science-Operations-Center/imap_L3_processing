from datetime import timedelta

import numpy as np
from imap_data_access import upload

from imap_l3_processing import spice_wrapper
from imap_l3_processing.data_utils import find_closest_neighbor
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import calculate_solar_wind_velocity_vector
from imap_l3_processing.swe.l3.models import SweL3Data, SweL1bData, SweL2Data, SweL3MomentData
from imap_l3_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    rotate_temperature, \
    rotate_dps_vector_to_rtn, core_fit_moments_retrying_on_failure, halo_fit_moments_retrying_on_failure
from imap_l3_processing.swe.l3.science.pitch_calculations import average_over_look_directions, find_breakpoints, \
    correct_and_rebin, \
    integrate_distribution_to_get_1d_spectrum, integrate_distribution_to_get_inbound_and_outbound_1d_spectrum, \
    calculate_velocity_in_dsp_frame_km_s
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.utils import save_data


class SweProcessor(Processor):
    def process(self):
        dependencies = SweL3Dependencies.fetch_dependencies(self.dependencies)
        output_data = self.calculate_products(dependencies)
        output_cdf = save_data(output_data)
        upload(output_cdf)

    def calculate_products(self, dependencies: SweL3Dependencies) -> SweL3Data:
        spice_wrapper.furnish()
        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
        config = dependencies.configuration

        spacecraft_potential_history = [config["spacecraft_potential_initial_guess"] for _ in
                                        range(3)]
        halo_core_history = [config["core_halo_breakpoint_initial_guess"] for _ in range(3)]

        average_psd = []
        spacecraft_potential: np.ndarray[np.float64] = np.empty_like(swe_epoch, dtype=np.float64)
        halo_core: np.ndarray[np.float64] = np.empty_like(swe_epoch, dtype=np.float64)
        corrected_energy_bins = []

        for i in range(len(swe_epoch)):
            average_psd.append(average_over_look_directions(swe_l2_data.phase_space_density[i],
                                                            np.array(config["geometric_fractions"]),
                                                            config["minimum_phase_space_density_value"]))

            spacecraft_potential[i], halo_core[i] = find_breakpoints(swe_l2_data.energy, average_psd[i],
                                                                     spacecraft_potential_history,
                                                                     halo_core_history,
                                                                     config)

            spacecraft_potential_history = [*spacecraft_potential_history[1:], spacecraft_potential[i]]
            halo_core_history = [*halo_core_history[1:], halo_core[i]]
            corrected_energy_bins.append(swe_l2_data.energy - spacecraft_potential[i])

        corrected_energy_bins = np.array(corrected_energy_bins)

        swe_l3_moments_data = self.calculate_moment_products(swe_l2_data, dependencies.swe_l1b_data,
                                                             spacecraft_potential, halo_core,
                                                             corrected_energy_bins)

        phase_space_density_by_pitch_angle, energy_spectrum, energy_spectrum_inbound, energy_spectrum_outbound = self.calculate_pitch_angle_products(
            dependencies, corrected_energy_bins)

        return SweL3Data(
            input_metadata=self.input_metadata.to_upstream_data_dependency("sci"),
            epoch=swe_epoch,
            epoch_delta=swe_l2_data.epoch_delta,
            energy=config["energy_bins"],
            energy_delta_plus=config["energy_delta_plus"],
            energy_delta_minus=config["energy_delta_minus"],
            pitch_angle=config["pitch_angle_bins"],
            pitch_angle_delta=config["pitch_angle_delta"],
            spacecraft_potential=spacecraft_potential,
            core_halo_breakpoint=halo_core,
            phase_space_density_by_pitch_angle=phase_space_density_by_pitch_angle,
            energy_spectrum=energy_spectrum,
            energy_spectrum_inbound=energy_spectrum_inbound,
            energy_spectrum_outbound=energy_spectrum_outbound,
            core_fit_num_points=swe_l3_moments_data.core_fit_num_points,
            core_chisq=swe_l3_moments_data.core_chisq,
            halo_chisq=swe_l3_moments_data.halo_chisq,
            core_density_fit=swe_l3_moments_data.core_density_fit,
            halo_density_fit=swe_l3_moments_data.halo_density_fit,
            core_t_parallel_fit=swe_l3_moments_data.core_t_parallel_fit,
            halo_t_parallel_fit=swe_l3_moments_data.halo_t_parallel_fit,
            core_t_perpendicular_fit=swe_l3_moments_data.core_t_perpendicular_fit,
            halo_t_perpendicular_fit=swe_l3_moments_data.halo_t_perpendicular_fit,
            core_temperature_phi_rtn_fit=swe_l3_moments_data.core_temperature_phi_rtn_fit,
            halo_temperature_phi_rtn_fit=swe_l3_moments_data.halo_temperature_phi_rtn_fit,
            core_temperature_theta_rtn_fit=swe_l3_moments_data.core_temperature_theta_rtn_fit,
            halo_temperature_theta_rtn_fit=swe_l3_moments_data.halo_temperature_theta_rtn_fit,
            core_speed_fit=swe_l3_moments_data.core_speed_fit,
            halo_speed_fit=swe_l3_moments_data.halo_speed_fit,
            core_velocity_vector_rtn_fit=swe_l3_moments_data.core_velocity_vector_rtn_fit,
            halo_velocity_vector_rtn_fit=swe_l3_moments_data.halo_velocity_vector_rtn_fit,
        )

    def calculate_moment_products(self, swe_l2_data: SweL2Data, swe_l1b_data: SweL1bData,
                                  spacecraft_potential: np.ndarray, halo_core: np.ndarray,
                                  corrected_energy_bins: np.ndarray):

        core_density_history = [100 for _ in range(3)]
        halo_density_history = [25 for _ in range(3)]
        core_moments = []
        core_fit_chi_squareds = []
        core_fit_num_points = []
        halo_moments = []
        halo_fit_chi_squareds = []
        core_rtn_velocity = []
        halo_rtn_velocity = []
        core_temp_theta_rtns = []
        core_temp_phi_rtns = []
        halo_temp_theta_rtns = []
        halo_temp_phi_rtns = []

        for i in range(len(swe_l2_data.epoch)):
            velocity_vectors: np.ndarray = calculate_velocity_in_dsp_frame_km_s(corrected_energy_bins[i],
                                                                                swe_l2_data.inst_el,
                                                                                swe_l2_data.inst_az_spin_sector[i])

            weights: np.ndarray[float] = compute_maxwellian_weight_factors(swe_l1b_data.count_rates[i],
                                                                           swe_l2_data.acquisition_duration[i])

            spacecraft_potential_core_breakpoint_index: int = next(
                energy_i for energy_i, energy in enumerate(swe_l2_data.energy) if energy >= spacecraft_potential[i])
            halo_core_breakpoint_index: int = next(
                energy_i - 1 for energy_i, energy in enumerate(swe_l2_data.energy) if energy > halo_core[i])

            core_end_index = halo_core_breakpoint_index

            core_moment_fit_result = core_fit_moments_retrying_on_failure(
                corrected_energy_bins[i],
                velocity_vectors,
                swe_l2_data.phase_space_density[i],
                weights,
                spacecraft_potential_core_breakpoint_index,
                core_end_index,
                core_density_history
            )
            core_moment = core_moment_fit_result.moments
            core_moments.append(core_moment)
            core_fit_chi_squareds.append(core_moment_fit_result.chisq)
            core_fit_num_points.append(core_moment_fit_result.number_of_points)

            core_density_history = [*core_density_history[1:], core_moment.density]

            core_rtn_velocity.append(rotate_dps_vector_to_rtn(swe_l2_data.epoch[i],
                                                              np.array(
                                                                  [core_moment.velocity_x, core_moment.velocity_y,
                                                                   core_moment.velocity_z])))

            core_temp_theta_rtn, core_temp_phi_rtn = rotate_temperature(swe_l2_data.epoch[i], core_moment.alpha,
                                                                        core_moment.beta)
            core_temp_theta_rtns.append(core_temp_theta_rtn)
            core_temp_phi_rtns.append(core_temp_phi_rtn)

            halo_end_index = len(swe_l2_data.energy)
            halo_moment_fit_result = halo_fit_moments_retrying_on_failure(
                corrected_energy_bins[i],
                velocity_vectors,
                swe_l2_data.phase_space_density[i],
                weights,
                halo_core_breakpoint_index,
                halo_end_index,
                halo_density_history,
                spacecraft_potential[i],
                halo_core[i],
            )
            halo_moment = halo_moment_fit_result.moments

            halo_moments.append(halo_moment)
            halo_fit_chi_squareds.append(halo_moment_fit_result.chisq)

            halo_density_history = [*halo_density_history[1:], halo_moment.density]

            halo_rtn_velocity.append(rotate_dps_vector_to_rtn(swe_l2_data.epoch[i],
                                                              np.array(
                                                                  [halo_moment.velocity_x, halo_moment.velocity_y,
                                                                   halo_moment.velocity_z])))

            halo_temp_theta_rtn, halo_temp_phi_rtn = rotate_temperature(swe_l2_data.epoch[i], halo_moment.alpha,
                                                                        halo_moment.beta)
            halo_temp_theta_rtns.append(halo_temp_theta_rtn)
            halo_temp_phi_rtns.append(halo_temp_phi_rtn)

        return SweL3MomentData(
            core_fit_num_points=np.array(core_fit_num_points),
            core_chisq=np.array(core_fit_chi_squareds),
            halo_chisq=np.array(halo_fit_chi_squareds),
            core_density_fit=np.array([core_moment.density for core_moment in core_moments]),
            halo_density_fit=np.array([halo_moment.density for halo_moment in halo_moments]),
            core_t_parallel_fit=np.array([core_moment.t_parallel for core_moment in core_moments]),
            halo_t_parallel_fit=np.array([halo_moment.t_parallel for halo_moment in halo_moments]),
            core_t_perpendicular_fit=np.array([core_moment.t_perpendicular for core_moment in core_moments]),
            halo_t_perpendicular_fit=np.array([halo_moment.t_perpendicular for halo_moment in halo_moments]),
            core_temperature_phi_rtn_fit=np.array(core_temp_phi_rtns),
            halo_temperature_phi_rtn_fit=np.array(halo_temp_phi_rtns),
            core_temperature_theta_rtn_fit=np.array(core_temp_theta_rtns),
            halo_temperature_theta_rtn_fit=np.array(halo_temp_theta_rtns),
            core_speed_fit=np.linalg.norm(np.array(core_rtn_velocity), axis=-1),
            halo_speed_fit=np.linalg.norm(np.array(halo_rtn_velocity), axis=-1),
            core_velocity_vector_rtn_fit=np.array(core_rtn_velocity),
            halo_velocity_vector_rtn_fit=np.array(halo_rtn_velocity),
        )

    def calculate_pitch_angle_products(self, dependencies: SweL3Dependencies, corrected_energy_bins: np.ndarray):
        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
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

        for i in range(len(swe_epoch)):
            missing_mag_data = np.any(np.isnan(rebinned_mag_data[i]))
            if missing_mag_data:
                num_energy_bins = len(config['energy_bins'])
                num_pitch_angle_bins = len(config['pitch_angle_bins'])
                phase_space_density_by_pitch_angle.append(np.full((num_energy_bins, num_pitch_angle_bins), np.nan))
                energy_spectrum.append(np.full(num_energy_bins, np.nan))
                energy_spectrum_inbound.append(np.full(num_energy_bins, np.nan))
                energy_spectrum_outbound.append(np.full(num_energy_bins, np.nan))
            else:

                rebinned_psd = correct_and_rebin(swe_l2_data.phase_space_density[i], corrected_energy_bins[i],
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

        return phase_space_density_by_pitch_angle, energy_spectrum, energy_spectrum_inbound, energy_spectrum_outbound


def calculate_pitch(arg1, arg_1, arg_3):
    pass
