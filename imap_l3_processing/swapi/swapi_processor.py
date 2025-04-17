import imap_data_access
import numpy as np
from uncertainties.unumpy import uarray, nominal_values

from imap_l3_processing import spice_wrapper
from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.processor import Processor
from imap_l3_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, SwapiL3AlphaSolarWindData, \
    SwapiL3PickupIonData
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed, \
    calculate_combined_sweeps
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import calculate_ten_minute_velocities, \
    calculate_pickup_ion_values, calculate_helium_pui_temperature, calculate_helium_pui_density
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_deflection_angle, calculate_clock_angle
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    calculate_proton_solar_wind_temperature_and_density
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data, chunk_l2_data
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import \
    calculate_combined_solar_wind_differential_flux
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    calculate_alpha_solar_wind_vdf, calculate_pui_solar_wind_vdf, calculate_delta_minus_plus
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.utils import save_data


class SwapiProcessor(Processor):

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = SwapiL3ADependencies.fetch_dependencies(self.dependencies)
            data = read_l2_swapi_data(l3a_dependencies.data)
            self.get_parent_file_names()
            proton_data, alpha_data, pui_he_data = self.process_l3a(data, l3a_dependencies)

            if self.input_metadata.descriptor == 'proton-sw':
                proton_cdf = save_data(proton_data)
                imap_data_access.upload(proton_cdf)
            elif self.input_metadata.descriptor == 'alpha-sw':
                alpha_cdf = save_data(alpha_data)
                imap_data_access.upload(alpha_cdf)
            elif self.input_metadata.descriptor == 'pui-he':
                pui_he_cdf = save_data(pui_he_data)
                imap_data_access.upload(pui_he_cdf)
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(self.dependencies)
            data = read_l2_swapi_data(l3b_dependencies.data)
            l3b_combined_vdf = self.process_l3b(data, l3b_dependencies)
            cdf_path = save_data(l3b_combined_vdf)
            imap_data_access.upload(cdf_path)

    def process_l3a(self, data, dependencies) -> tuple[
        SwapiL3ProtonSolarWindData, SwapiL3AlphaSolarWindData, SwapiL3PickupIonData]:
        spice_wrapper.furnish()
        epochs = []

        proton_solar_wind_speeds = []
        proton_solar_wind_temperatures = []
        proton_solar_wind_density = []
        proton_solar_wind_clock_angles = []
        proton_solar_wind_deflection_angles = []

        alpha_solar_wind_speeds = []
        alpha_solar_wind_densities = []
        alpha_solar_wind_temperatures = []

        for data_chunk in chunk_l2_data(data, 5):
            coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                              data_chunk.coincidence_count_rate_uncertainty)
            proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                coincidence_count_rates_with_uncertainty, data_chunk.energy, data_chunk.epoch)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)

            clock_angle = calculate_clock_angle(dependencies.clock_angle_and_flow_deflection_calibration_table,
                                                proton_solar_wind_speed, a, phi, b)

            deflection_angle = calculate_deflection_angle(
                dependencies.clock_angle_and_flow_deflection_calibration_table,
                proton_solar_wind_speed, a, phi, b)

            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)

            proton_temperature, proton_density = calculate_proton_solar_wind_temperature_and_density(
                dependencies.proton_temperature_density_calibration_table,
                proton_solar_wind_speed,
                deflection_angle,
                clock_angle,
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy)

            proton_solar_wind_temperatures.append(proton_temperature)
            proton_solar_wind_density.append(proton_density)

            epochs.append(data_chunk.epoch[0] + THIRTY_SECONDS_IN_NANOSECONDS)

            alpha_solar_wind_speed = calculate_alpha_solar_wind_speed(coincidence_count_rates_with_uncertainty,
                                                                      data_chunk.energy)
            alpha_solar_wind_speeds.append(alpha_solar_wind_speed)

            alpha_temperature, alpha_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
                dependencies.alpha_temperature_density_calibration_table, alpha_solar_wind_speed,
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy)

            alpha_solar_wind_densities.append(alpha_density)
            alpha_solar_wind_temperatures.append(alpha_temperature)

        ten_minute_solar_wind_velocities = calculate_ten_minute_velocities(
            nominal_values(proton_solar_wind_speeds),
            nominal_values(proton_solar_wind_deflection_angles),
            nominal_values(proton_solar_wind_clock_angles))
        pui_epochs = []
        pui_cooling_index = []
        pui_ionization_rate = []
        pui_cutoff_speed = []
        pui_background_rate = []
        pui_density = []
        pui_temperature = []
        for data_chunk, sw_velocity in zip(chunk_l2_data(data, 50), ten_minute_solar_wind_velocities):
            epoch = data_chunk.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS
            fit_params = calculate_pickup_ion_values(dependencies.instrument_response_calibration_table,
                                                     dependencies.geometric_factor_calibration_table, data_chunk.energy,
                                                     data_chunk.coincidence_count_rate,
                                                     epoch, 0.1,
                                                     sw_velocity,
                                                     dependencies.density_of_neutral_helium_calibration_table)
            pui_epochs.append(epoch)
            pui_cooling_index.append(fit_params.cooling_index)
            pui_ionization_rate.append(fit_params.ionization_rate)
            pui_cutoff_speed.append(fit_params.cutoff_speed)
            pui_background_rate.append(fit_params.background_count_rate)
            density = calculate_helium_pui_density(
                epoch, sw_velocity, dependencies.density_of_neutral_helium_calibration_table, fit_params)
            pui_density.append(density)
            temperature = calculate_helium_pui_temperature(
                epoch, sw_velocity, dependencies.density_of_neutral_helium_calibration_table, fit_params)
            pui_temperature.append(temperature)
        pui_metadata = self.input_metadata.to_upstream_data_dependency("pui-he")
        pui_data = SwapiL3PickupIonData(pui_metadata, np.array(pui_epochs), np.array(pui_cooling_index),
                                        np.array(pui_ionization_rate),
                                        np.array(pui_cutoff_speed), np.array(pui_background_rate),
                                        np.array(pui_density), np.array(pui_temperature))

        proton_solar_wind_speed_metadata = self.input_metadata.to_upstream_data_dependency("proton-sw")

        proton_solar_wind_l3_data = SwapiL3ProtonSolarWindData(proton_solar_wind_speed_metadata, np.array(epochs),
                                                               np.array(proton_solar_wind_speeds),
                                                               np.array(proton_solar_wind_temperatures),
                                                               np.array(proton_solar_wind_density),
                                                               np.array(proton_solar_wind_clock_angles),
                                                               np.array(proton_solar_wind_deflection_angles))

        alpha_solar_wind_speed_metadata = self.input_metadata.to_upstream_data_dependency("alpha-sw")
        alpha_solar_wind_l3_data = SwapiL3AlphaSolarWindData(alpha_solar_wind_speed_metadata, np.array(epochs),
                                                             np.array(alpha_solar_wind_speeds),
                                                             np.array(alpha_solar_wind_temperatures),
                                                             np.array(alpha_solar_wind_densities))
        return proton_solar_wind_l3_data, alpha_solar_wind_l3_data, pui_data

    def process_l3b(self, data, dependencies):
        epochs = []
        cdf_proton_velocities = []
        cdf_proton_probabilities = []
        cdf_alpha_velocities = []
        cdf_alpha_probabilities = []
        cdf_pui_velocities = []
        cdf_pui_probabilities = []
        combined_differential_fluxes = []
        combined_energies = []
        cdf_proton_deltas = []
        cdf_alpha_deltas = []
        cdf_pui_deltas = []
        combined_energy_deltas = []
        for data_chunk in chunk_l2_data(data, 50):
            center_of_epoch = data_chunk.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS
            instrument_efficiency = dependencies.efficiency_calibration_table.get_efficiency_for(center_of_epoch)
            coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                              data_chunk.coincidence_count_rate_uncertainty)

            average_coincident_count_rates, energies = calculate_combined_sweeps(
                coincidence_count_rates_with_uncertainty, data_chunk.energy)

            proton_velocities, proton_probabilities = calculate_proton_solar_wind_vdf(energies,
                                                                                      average_coincident_count_rates,
                                                                                      instrument_efficiency,
                                                                                      dependencies.geometric_factor_calibration_table)

            alpha_velocities, alpha_probabilities = calculate_alpha_solar_wind_vdf(energies,
                                                                                   average_coincident_count_rates,
                                                                                   instrument_efficiency,
                                                                                   dependencies.geometric_factor_calibration_table)

            pui_velocities, pui_probabilities = calculate_pui_solar_wind_vdf(energies,
                                                                             average_coincident_count_rates,
                                                                             instrument_efficiency,
                                                                             dependencies.geometric_factor_calibration_table)
            combined_differential_flux = calculate_combined_solar_wind_differential_flux(energies,
                                                                                         average_coincident_count_rates,
                                                                                         instrument_efficiency,
                                                                                         dependencies.geometric_factor_calibration_table)

            epochs.append(center_of_epoch)
            cdf_proton_velocities.append(proton_velocities)
            cdf_proton_probabilities.append(proton_probabilities)
            cdf_proton_deltas.append(calculate_delta_minus_plus(proton_velocities))

            cdf_alpha_velocities.append(alpha_velocities)
            cdf_alpha_probabilities.append(alpha_probabilities)
            cdf_alpha_deltas.append(calculate_delta_minus_plus(alpha_velocities))

            cdf_pui_velocities.append(pui_velocities)
            cdf_pui_probabilities.append(pui_probabilities)
            cdf_pui_deltas.append(calculate_delta_minus_plus(pui_velocities))

            combined_differential_fluxes.append(combined_differential_flux)
            combined_energies.append(energies)
            combined_energy_deltas.append(calculate_delta_minus_plus(energies))

        l3b_combined_metadata = self.input_metadata.to_upstream_data_dependency("combined")
        l3b_combined_vdf = SwapiL3BCombinedVDF(input_metadata=l3b_combined_metadata,
                                               epoch=np.array(epochs),
                                               proton_sw_velocities=np.array(cdf_proton_velocities),
                                               proton_sw_velocities_delta_minus=np.array(
                                                   [delta.delta_minus for delta in cdf_proton_deltas]),
                                               proton_sw_velocities_delta_plus=np.array(
                                                   [delta.delta_plus for delta in cdf_proton_deltas]),
                                               proton_sw_combined_vdf=np.array(cdf_proton_probabilities),
                                               alpha_sw_velocities=np.array(cdf_alpha_velocities),
                                               alpha_sw_velocities_delta_minus=np.array(
                                                   [delta.delta_minus for delta in cdf_alpha_deltas]),
                                               alpha_sw_velocities_delta_plus=np.array(
                                                   [delta.delta_plus for delta in cdf_alpha_deltas]),
                                               alpha_sw_combined_vdf=np.array(cdf_alpha_probabilities),
                                               pui_sw_velocities=np.array(cdf_pui_velocities),
                                               pui_sw_velocities_delta_minus=np.array(
                                                   [delta.delta_minus for delta in cdf_pui_deltas]),
                                               pui_sw_velocities_delta_plus=np.array(
                                                   [delta.delta_plus for delta in cdf_pui_deltas]),
                                               pui_sw_combined_vdf=np.array(cdf_pui_probabilities),
                                               combined_energy=np.array(combined_energies),
                                               combined_energy_delta_minus=np.array(
                                                   [delta.delta_minus for delta in combined_energy_deltas]),
                                               combined_energy_delta_plus=np.array(
                                                   [delta.delta_plus for delta in combined_energy_deltas]),
                                               combined_differential_flux=np.array(combined_differential_fluxes),
                                               )
        return l3b_combined_vdf
