from datetime import timedelta

import numpy as np
from imap_data_access import upload

from imap_l3_processing import spice_wrapper
from imap_l3_processing.data_utils import find_closest_neighbor
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import calculate_solar_wind_velocity_vector
from imap_l3_processing.swe.l3.models import SweL3Data, SweL1bData, SweL2Data, SweL3MomentData, SweConfiguration
from imap_l3_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    rotate_temperature, \
    rotate_dps_vector_to_rtn, core_fit_moments_retrying_on_failure, halo_fit_moments_retrying_on_failure, Moments, \
    integrate, scale_core_density, scale_halo_density, rotate_vector_to_rtn_spherical_coordinates, \
    calculate_primary_eigenvector, \
    ScaleDensityOutput, rotate_temperature_tensor_to_mag
from imap_l3_processing.swe.l3.science.pitch_calculations import average_over_look_directions, find_breakpoints, \
    correct_and_rebin, \
    integrate_distribution_to_get_1d_spectrum, integrate_distribution_to_get_inbound_and_outbound_1d_spectrum, \
    calculate_velocity_in_dsp_frame_km_s
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.l3.utils import compute_epoch_delta_in_ns
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
        epoch_delta = compute_epoch_delta_in_ns(swe_l2_data.acquisition_duration,
                                                dependencies.swe_l1b_data.settle_duration)
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
                                                             corrected_energy_bins, config)

        phase_space_density_by_pitch_angle, energy_spectrum, energy_spectrum_inbound, energy_spectrum_outbound = self.calculate_pitch_angle_products(
            dependencies, corrected_energy_bins)

        return SweL3Data(
            input_metadata=self.input_metadata.to_upstream_data_dependency("sci"),
            epoch=swe_epoch,
            epoch_delta=epoch_delta,
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
            moment_data=swe_l3_moments_data
        )

    def calculate_moment_products(self, swe_l2_data: SweL2Data, swe_l1b_data: SweL1bData,
                                  spacecraft_potential: np.ndarray, halo_core: np.ndarray,
                                  corrected_energy_bins: np.ndarray, config: SweConfiguration) -> SweL3MomentData:
        number_of_points = len(swe_l2_data.epoch)
        core_density_history = [100 for _ in range(3)]
        halo_density_history = [25 for _ in range(3)]
        core_moments = np.full(number_of_points, Moments.construct_all_fill())
        core_fit_chi_squareds = np.full(number_of_points, np.nan)
        core_fit_num_points = np.full(number_of_points, np.nan)
        halo_moments = np.full(number_of_points, Moments.construct_all_fill())
        halo_fit_chi_squareds = np.full(number_of_points, np.nan)
        core_rtn_velocity = np.full((number_of_points, 3), np.nan)
        halo_rtn_velocity = np.full((number_of_points, 3), np.nan)
        core_temp_theta_rtns = np.full(number_of_points, np.nan)
        core_temp_phi_rtns = np.full(number_of_points, np.nan)
        halo_temp_theta_rtns = np.full(number_of_points, np.nan)
        halo_temp_phi_rtns = np.full(number_of_points, np.nan)
        core_density_integrated = np.full(number_of_points, np.nan)
        halo_density_integrated = np.full(number_of_points, np.nan)
        total_density_integrated = np.full(number_of_points, np.nan)
        core_velocity_integrated = np.full((number_of_points, 3), np.nan)
        halo_velocity_integrated = np.full((number_of_points, 3), np.nan)
        total_velocity_integrated = np.full((number_of_points, 3), np.nan)
        core_heat_flux_magnitude = np.full(number_of_points, np.nan)
        core_heat_flux_theta = np.full(number_of_points, np.nan)
        core_heat_flux_phi = np.full(number_of_points, np.nan)
        halo_heat_flux_magnitude = np.full(number_of_points, np.nan)
        halo_heat_flux_theta = np.full(number_of_points, np.nan)
        halo_heat_flux_phi = np.full(number_of_points, np.nan)
        total_heat_flux_magnitude = np.full(number_of_points, np.nan)
        total_heat_flux_theta = np.full(number_of_points, np.nan)
        total_heat_flux_phi = np.full(number_of_points, np.nan)
        core_t_parallel_integrated = np.full(number_of_points, np.nan)
        core_t_perpendicular_integrated = np.full((number_of_points, 2), np.nan)
        halo_t_parallel_integrated = np.full(number_of_points, np.nan)
        halo_t_perpendicular_integrated = np.full((number_of_points, 2), np.nan)
        total_t_parallel_integrated = np.full(number_of_points, np.nan)
        total_t_perpendicular_integrated = np.full((number_of_points, 2), np.nan)
        core_temperature_theta_rtn_integrated = np.full(number_of_points, np.nan)
        core_temperature_phi_rtn_integrated = np.full(number_of_points, np.nan)
        halo_temperature_theta_rtn_integrated = np.full(number_of_points, np.nan)
        halo_temperature_phi_rtn_integrated = np.full(number_of_points, np.nan)
        total_temperature_theta_rtn_integrated = np.full(number_of_points, np.nan)
        total_temperature_phi_rtn_integrated = np.full(number_of_points, np.nan)
        core_temperature_parallel_to_mag = np.full(number_of_points, np.nan)
        core_temperature_perpendicular_to_mag = np.full((number_of_points, 2), np.nan)
        halo_temperature_parallel_to_mag = np.full(number_of_points, np.nan)
        halo_temperature_perpendicular_to_mag = np.full((number_of_points, 2), np.nan)
        total_temperature_parallel_to_mag = np.full(number_of_points, np.nan)
        total_temperature_perpendicular_to_mag = np.full((number_of_points, 2), np.nan)
        core_temperature_tensor_integrated = np.full((number_of_points, 6), np.nan)
        halo_temperature_tensor_integrated = np.full((number_of_points, 6), np.nan)
        total_temperature_tensor_integrated = np.full((number_of_points, 6), np.nan)

        for i in range(len(swe_l2_data.epoch)):
            velocity_vectors_cm_per_s: np.ndarray = 1000 * 100 * calculate_velocity_in_dsp_frame_km_s(
                corrected_energy_bins[i],
                swe_l2_data.inst_el,
                swe_l2_data.inst_az_spin_sector[
                    i])

            weights: np.ndarray[float] = compute_maxwellian_weight_factors(swe_l1b_data.count_rates[i],
                                                                           swe_l2_data.acquisition_duration[i])

            ifit = next(
                index for index, energy in enumerate(swe_l2_data.energy) if energy >= spacecraft_potential[i])
            jbreak = next(index for index, energy in enumerate(swe_l2_data.energy) if energy >= halo_core[i])
            core_nfit = jbreak - ifit
            ifit += 1

            halo_nfit = 5 if len(swe_l2_data.energy) - jbreak > 5 else len(swe_l2_data.energy) - jbreak

            core_moment_fit_result = core_fit_moments_retrying_on_failure(
                corrected_energy_bins[i],
                velocity_vectors_cm_per_s,
                swe_l2_data.phase_space_density[i],
                weights,
                ifit,
                ifit + core_nfit,
                core_density_history
            )

            current_epoch = swe_l2_data.epoch[i]
            if core_moment_fit_result is not None:
                core_moment = core_moment_fit_result.moments
                core_moments[i] = core_moment
                core_fit_chi_squareds[i] = core_moment_fit_result.chisq
                core_fit_num_points[i] = core_moment_fit_result.number_of_points

                core_density_history = [*core_density_history[1:], core_moment.density]

                core_rtn_velocity[i] = rotate_dps_vector_to_rtn(current_epoch,
                                                                np.array(
                                                                    [core_moment.velocity_x, core_moment.velocity_y,
                                                                     core_moment.velocity_z]))

                core_temp_theta_rtn, core_temp_phi_rtn = rotate_temperature(current_epoch, core_moment.alpha,
                                                                            core_moment.beta)
                core_temp_theta_rtns[i] = core_temp_theta_rtn
                core_temp_phi_rtns[i] = core_temp_phi_rtn
                sin_theta = np.sin(np.deg2rad(swe_l2_data.inst_el))
                cos_theta = np.cos(np.deg2rad(swe_l2_data.inst_el))
                core_temp_avg = (2 * core_moment.t_perpendicular + core_moment.t_parallel) / 3

                if 1e3 < core_temp_avg < 1e7:
                    core_integrate_result = integrate(ifit + 1, jbreak - 1, corrected_energy_bins[i],
                                                      sin_theta, cos_theta, config['aperture_field_of_view_radians'],
                                                      swe_l2_data.phase_space_density[i],
                                                      swe_l2_data.inst_az_spin_sector[i],
                                                      spacecraft_potential[i],
                                                      [0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
                    if core_integrate_result is not None:
                        scale_core_density_output: ScaleDensityOutput = scale_core_density(
                            core_integrate_result.density,
                            core_integrate_result.velocity,
                            core_integrate_result.temperature, core_moment,
                            ifit,
                            corrected_energy_bins[i],
                            spacecraft_potential[i], cos_theta,
                            config['aperture_field_of_view_radians'],
                            swe_l2_data.inst_az_spin_sector[i],
                            core_moment_fit_result.regress_result,
                            core_integrate_result.base_energy)

                        core_density_integrated[i] = scale_core_density_output.density
                        core_velocity_integrated[i] = rotate_dps_vector_to_rtn(current_epoch,
                                                                               scale_core_density_output.velocity)
                        core_temperature_tensor_integrated[i] = scale_core_density_output.temperature

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(current_epoch,
                                                                                           core_integrate_result.heat_flux)

                        core_heat_flux_magnitude[i] = magnitude
                        core_heat_flux_theta[i] = theta
                        core_heat_flux_phi[i] = phi

                        core_primary_eigen_vector, core_temps = calculate_primary_eigenvector(
                            scale_core_density_output.temperature)

                        core_t_parallel_integrated[i] = core_temps[0]
                        core_t_perpendicular_integrated[i] = [core_temps[1], core_temps[2]]

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(current_epoch,
                                                                                           core_primary_eigen_vector)
                        core_temperature_theta_rtn_integrated[i] = theta
                        core_temperature_phi_rtn_integrated[i] = phi

                        core_t_parallel_to_mag, core_t_perpendicular_to_mag_average, core_t_perpendicular_to_mag_ratio = rotate_temperature_tensor_to_mag(
                            scale_core_density_output.temperature, [0, 0, 1])

                        core_temperature_parallel_to_mag[i] = core_t_parallel_to_mag
                        core_temperature_perpendicular_to_mag[i] = [core_t_perpendicular_to_mag_average,
                                                                    core_t_perpendicular_to_mag_ratio]

                        total_integration_output = integrate(ifit + 1,
                                                             len(corrected_energy_bins[i]) - 1,
                                                             corrected_energy_bins[i],
                                                             sin_theta,
                                                             cos_theta, config['aperture_field_of_view_radians'],
                                                             swe_l2_data.phase_space_density[i],
                                                             swe_l2_data.inst_az_spin_sector[i],
                                                             spacecraft_potential[i], scale_core_density_output.cdelnv,
                                                             scale_core_density_output.cdelt)
                        assert total_integration_output is not None, "not yet checking if this is None"
                        total_density_integrated[i] = total_integration_output.density
                        total_velocity_integrated[i] = rotate_dps_vector_to_rtn(current_epoch,
                                                                                total_integration_output.velocity)
                        total_temperature_tensor_integrated[i] = total_integration_output.temperature

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(
                            current_epoch,
                            total_integration_output.heat_flux)

                        total_heat_flux_magnitude[i] = magnitude
                        total_heat_flux_theta[i] = theta
                        total_heat_flux_phi[i] = phi

                        total_primary_eigen_vector, total_temps = calculate_primary_eigenvector(
                            total_integration_output.temperature)

                        total_t_parallel_integrated[i] = total_temps[0]
                        total_t_perpendicular_integrated[i] = [total_temps[1], total_temps[2]]

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(current_epoch,
                                                                                           total_primary_eigen_vector)
                        total_temperature_theta_rtn_integrated[i] = theta
                        total_temperature_phi_rtn_integrated[i] = phi

                        total_t_parallel_to_mag, total_t_perpendicular_to_mag_average, total_t_perpendicular_to_mag_ratio = rotate_temperature_tensor_to_mag(
                            total_integration_output.temperature, [0, 0, 1])

                        total_temperature_parallel_to_mag[i] = total_t_parallel_to_mag
                        total_temperature_perpendicular_to_mag[i] = [total_t_perpendicular_to_mag_average,
                                                                     total_t_perpendicular_to_mag_ratio]

            halo_moment_fit_result = halo_fit_moments_retrying_on_failure(
                corrected_energy_bins[i],
                velocity_vectors_cm_per_s,
                swe_l2_data.phase_space_density[i],
                weights,
                jbreak,
                jbreak + halo_nfit,
                halo_density_history,
                spacecraft_potential[i],
                halo_core[i],
            )

            if halo_moment_fit_result is not None:
                halo_moment = halo_moment_fit_result.moments
                halo_moments[i] = halo_moment
                halo_fit_chi_squareds[i] = halo_moment_fit_result.chisq

                halo_density_history = [*halo_density_history[1:], halo_moment.density]

                halo_rtn_velocity[i] = rotate_dps_vector_to_rtn(current_epoch,
                                                                np.array(
                                                                    [halo_moment.velocity_x, halo_moment.velocity_y,
                                                                     halo_moment.velocity_z]))

                halo_temp_theta_rtn, halo_temp_phi_rtn = rotate_temperature(current_epoch, halo_moment.alpha,
                                                                            halo_moment.beta)
                halo_temp_theta_rtns[i] = halo_temp_theta_rtn
                halo_temp_phi_rtns[i] = halo_temp_phi_rtn
                halo_temp_avg = (2 * halo_moment.t_perpendicular + halo_moment.t_parallel) / 3
                if 1e4 < halo_temp_avg < 1e8:
                    halo_integrate_result = integrate(jbreak, len(corrected_energy_bins[i]) - 1,
                                                      corrected_energy_bins[i],
                                                      sin_theta, cos_theta, config['aperture_field_of_view_radians'],
                                                      swe_l2_data.phase_space_density[i],
                                                      swe_l2_data.inst_az_spin_sector[i],
                                                      spacecraft_potential[i],
                                                      [0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
                    if halo_integrate_result is not None:
                        scale_halo_density_output: ScaleDensityOutput = scale_halo_density(
                            halo_integrate_result.density,
                            halo_integrate_result.velocity,
                            halo_integrate_result.temperature, halo_moment,
                            spacecraft_potential[i], halo_core[i], cos_theta,
                            config['aperture_field_of_view_radians'],
                            swe_l2_data.inst_az_spin_sector[i],
                            halo_moment_fit_result.regress_result,
                            halo_integrate_result.base_energy)

                        halo_density_integrated[i] = scale_halo_density_output.density
                        halo_velocity_integrated[i] = rotate_dps_vector_to_rtn(current_epoch,
                                                                               scale_halo_density_output.velocity)
                        halo_temperature_tensor_integrated[i] = scale_halo_density_output.temperature

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(current_epoch,
                                                                                           halo_integrate_result.heat_flux)

                        halo_heat_flux_magnitude[i] = magnitude
                        halo_heat_flux_theta[i] = theta
                        halo_heat_flux_phi[i] = phi

                        halo_primary_eigen_vector, halo_temps = calculate_primary_eigenvector(
                            scale_halo_density_output.temperature)

                        halo_t_parallel_integrated[i] = halo_temps[0]
                        halo_t_perpendicular_integrated[i] = [halo_temps[1], halo_temps[2]]

                        magnitude, theta, phi = rotate_vector_to_rtn_spherical_coordinates(current_epoch,
                                                                                           halo_primary_eigen_vector)
                        halo_temperature_theta_rtn_integrated[i] = theta
                        halo_temperature_phi_rtn_integrated[i] = phi

                        halo_t_parallel_to_mag, halo_t_perpendicular_to_mag_average, halo_t_perpendicular_to_mag_ratio = rotate_temperature_tensor_to_mag(
                            scale_halo_density_output.temperature, [0, 0, 1])

                        halo_temperature_parallel_to_mag[i] = halo_t_parallel_to_mag
                        halo_temperature_perpendicular_to_mag[i] = [halo_t_perpendicular_to_mag_average,
                                                                    halo_t_perpendicular_to_mag_ratio]

        return SweL3MomentData(
            core_fit_num_points=core_fit_num_points,
            core_chisq=core_fit_chi_squareds,
            halo_chisq=halo_fit_chi_squareds,
            core_density_fit=np.array([core_moment.density for core_moment in core_moments]),
            halo_density_fit=np.array([halo_moment.density for halo_moment in halo_moments]),
            core_t_parallel_fit=np.array([core_moment.t_parallel for core_moment in core_moments]),
            halo_t_parallel_fit=np.array([halo_moment.t_parallel for halo_moment in halo_moments]),
            core_t_perpendicular_fit=np.array([core_moment.t_perpendicular for core_moment in core_moments]),
            halo_t_perpendicular_fit=np.array([halo_moment.t_perpendicular for halo_moment in halo_moments]),
            core_temperature_phi_rtn_fit=core_temp_phi_rtns,
            halo_temperature_phi_rtn_fit=halo_temp_phi_rtns,
            core_temperature_theta_rtn_fit=core_temp_theta_rtns,
            halo_temperature_theta_rtn_fit=halo_temp_theta_rtns,
            core_speed_fit=np.linalg.norm(core_rtn_velocity, axis=-1),
            halo_speed_fit=np.linalg.norm(halo_rtn_velocity, axis=-1),
            core_velocity_vector_rtn_fit=core_rtn_velocity,
            halo_velocity_vector_rtn_fit=halo_rtn_velocity,
            core_density_integrated=core_density_integrated,
            halo_density_integrated=halo_density_integrated,
            total_density_integrated=total_density_integrated,
            core_speed_integrated=np.linalg.norm(core_velocity_integrated, axis=-1),
            halo_speed_integrated=np.linalg.norm(halo_velocity_integrated, axis=-1),
            total_speed_integrated=np.linalg.norm(total_velocity_integrated, axis=-1),
            core_velocity_vector_rtn_integrated=core_velocity_integrated,
            halo_velocity_vector_rtn_integrated=halo_velocity_integrated,
            total_velocity_vector_rtn_integrated=total_velocity_integrated,
            core_heat_flux_magnitude_integrated=core_heat_flux_magnitude,
            core_heat_flux_phi_integrated=core_heat_flux_phi,
            core_heat_flux_theta_integrated=core_heat_flux_theta,
            halo_heat_flux_magnitude_integrated=halo_heat_flux_magnitude,
            halo_heat_flux_phi_integrated=halo_heat_flux_phi,
            halo_heat_flux_theta_integrated=halo_heat_flux_theta,
            total_heat_flux_magnitude_integrated=total_heat_flux_magnitude,
            total_heat_flux_theta_integrated=total_heat_flux_theta,
            total_heat_flux_phi_integrated=total_heat_flux_phi,
            core_t_parallel_integrated=core_t_parallel_integrated,
            core_t_perpendicular_integrated=core_t_perpendicular_integrated,
            halo_t_parallel_integrated=halo_t_parallel_integrated,
            halo_t_perpendicular_integrated=halo_t_perpendicular_integrated,
            total_t_parallel_integrated=total_t_parallel_integrated,
            total_t_perpendicular_integrated=total_t_perpendicular_integrated,
            core_temperature_theta_rtn_integrated=core_temperature_theta_rtn_integrated,
            core_temperature_phi_rtn_integrated=core_temperature_phi_rtn_integrated,
            halo_temperature_theta_rtn_integrated=halo_temperature_theta_rtn_integrated,
            halo_temperature_phi_rtn_integrated=halo_temperature_phi_rtn_integrated,
            total_temperature_theta_rtn_integrated=total_temperature_theta_rtn_integrated,
            total_temperature_phi_rtn_integrated=total_temperature_phi_rtn_integrated,
            core_temperature_parallel_to_mag=core_temperature_parallel_to_mag,
            core_temperature_perpendicular_to_mag=core_temperature_perpendicular_to_mag,
            halo_temperature_parallel_to_mag=halo_temperature_parallel_to_mag,
            halo_temperature_perpendicular_to_mag=halo_temperature_perpendicular_to_mag,
            total_temperature_parallel_to_mag=total_temperature_parallel_to_mag,
            total_temperature_perpendicular_to_mag=total_temperature_perpendicular_to_mag,
            core_temperature_tensor_integrated=core_temperature_tensor_integrated,
            halo_temperature_tensor_integrated=halo_temperature_tensor_integrated,
            total_temperature_tensor_integrated=total_temperature_tensor_integrated,
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
                dsp_velocities = calculate_velocity_in_dsp_frame_km_s(corrected_energy_bins[i], swe_l2_data.inst_el,
                                                                      swe_l2_data.inst_az_spin_sector[i])
                rebinned_psd = correct_and_rebin(swe_l2_data.phase_space_density[i],
                                                 rebinned_solar_wind_vectors[i],
                                                 dsp_velocities,
                                                 rebinned_mag_data[i],
                                                 config)
                phase_space_density_by_pitch_angle.append(rebinned_psd)
                energy_spectrum.append(integrate_distribution_to_get_1d_spectrum(rebinned_psd, config))
                inbound, outbound = integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(rebinned_psd,
                                                                                                   config)
                energy_spectrum_inbound.append(inbound)
                energy_spectrum_outbound.append(outbound)

        return phase_space_density_by_pitch_angle, energy_spectrum, energy_spectrum_inbound, energy_spectrum_outbound
