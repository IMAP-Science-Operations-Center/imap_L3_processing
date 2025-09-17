import logging
from dataclasses import replace

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata
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
from imap_l3_processing.swapi.l3a.science.speed_calculation import extract_coarse_sweep
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import chunk_l2_data
from imap_l3_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_differential_flux import \
    calculate_combined_solar_wind_differential_flux
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    calculate_alpha_solar_wind_vdf, calculate_pui_solar_wind_vdf, calculate_delta_minus_plus
from imap_l3_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_l3_processing.utils import save_data

logger=logging.getLogger(__name__)

class SwapiProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = SwapiL3ADependencies.fetch_dependencies(self.dependencies)

            if self.input_metadata.descriptor == 'proton-sw':
                data = self.process_l3a_proton(l3a_dependencies.data, l3a_dependencies)
            elif self.input_metadata.descriptor == 'alpha-sw':
                data = self.process_l3a_alpha_solar_wind(l3a_dependencies.data, l3a_dependencies)
            elif self.input_metadata.descriptor == 'pui-he':
                data = self.process_l3a_pui(l3a_dependencies.data, l3a_dependencies)
            else:
                raise NotImplementedError("unknown descriptor", self.input_metadata.descriptor)
            data.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(data)
            return [cdf_path]
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(self.dependencies)
            l3b_combined_vdf = self.process_l3b(l3b_dependencies.data, l3b_dependencies)
            l3b_combined_vdf.parent_file_names = self.get_parent_file_names()
            cdf_path = save_data(l3b_combined_vdf)
            return [cdf_path]

    def process_l3a_pui(self, data, dependencies) -> SwapiL3PickupIonData:
        proton_solar_wind_speeds = []
        proton_solar_wind_clock_angles = []
        proton_solar_wind_deflection_angles = []

        for data_chunk in chunk_l2_data(data, 5):
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            proton_solar_wind_speed = ufloat(np.nan, np.nan)
            clock_angle = ufloat(np.nan, np.nan)
            deflection_angle = ufloat(np.nan, np.nan)
            try:
                coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                                  data_chunk.coincidence_count_rate_uncertainty)
                proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                    coincidence_count_rates_with_uncertainty, data_chunk.energy, data_chunk.sci_start_time)
                clock_angle = calculate_clock_angle(dependencies.clock_angle_and_flow_deflection_calibration_table,
                                                    proton_solar_wind_speed, a, phi, b)
                deflection_angle = calculate_deflection_angle(
                    dependencies.clock_angle_and_flow_deflection_calibration_table,
                    proton_solar_wind_speed, a, phi, b)
            except Exception as e:
                logger.info(f"Exception occurred at epoch {epoch}, continuing with fill value", exc_info=True)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)
            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)

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
            epoch = data_chunk.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
            cooling_index = ufloat(np.nan, np.nan)
            ionization_rate = ufloat(np.nan, np.nan)
            cutoff_speed = ufloat(np.nan, np.nan)
            background_count_rate = ufloat(np.nan, np.nan)
            density = ufloat(np.nan, np.nan)
            temperature = ufloat(np.nan, np.nan)
            try:
                if (np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate)))
                        or np.any(np.isnan(sw_velocity))):
                    raise ValueError("Fill values in input data")
                fit_params = calculate_pickup_ion_values(dependencies.instrument_response_calibration_table,
                                                         dependencies.geometric_factor_calibration_table, data_chunk.energy,
                                                         data_chunk.coincidence_count_rate,
                                                         epoch, 0.1,
                                                         sw_velocity,
                                                         dependencies.density_of_neutral_helium_calibration_table)
                cooling_index = fit_params.cooling_index
                ionization_rate = fit_params.ionization_rate
                cutoff_speed = fit_params.cutoff_speed
                background_count_rate = fit_params.background_count_rate

                density = calculate_helium_pui_density(
                    epoch, sw_velocity, dependencies.density_of_neutral_helium_calibration_table, fit_params)
                temperature = calculate_helium_pui_temperature(
                    epoch, sw_velocity, dependencies.density_of_neutral_helium_calibration_table, fit_params)
            except Exception as e:
                logger.info(f"Exception occurred at epoch {epoch}, continuing with fill value", exc_info=True)
            pui_epochs.append(epoch)
            pui_cooling_index.append(cooling_index)
            pui_ionization_rate.append(ionization_rate)
            pui_cutoff_speed.append(cutoff_speed)
            pui_background_rate.append(background_count_rate)
            pui_density.append(density)
            pui_temperature.append(temperature)

        pui_metadata = replace(self.input_metadata, descriptor="pui-he")
        pui_data = SwapiL3PickupIonData(pui_metadata, np.array(pui_epochs), np.array(pui_cooling_index),
                                        np.array(pui_ionization_rate),
                                        np.array(pui_cutoff_speed), np.array(pui_background_rate),
                                        np.array(pui_density), np.array(pui_temperature))

        return pui_data

    def process_l3a_alpha_solar_wind(self, data, dependencies) -> SwapiL3AlphaSolarWindData:
        epochs = []

        alpha_solar_wind_speeds = []
        alpha_solar_wind_densities = []
        alpha_solar_wind_temperatures = []

        for data_chunk in chunk_l2_data(data, 5):
            alpha_solar_wind_speed= ufloat(np.nan, np.nan)
            alpha_density= ufloat(np.nan, np.nan)
            alpha_temperature= ufloat(np.nan, np.nan)
            epoch = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
                    raise ValueError("Fill values in input data")
                coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                                  data_chunk.coincidence_count_rate_uncertainty)


                alpha_solar_wind_speed = calculate_alpha_solar_wind_speed(coincidence_count_rates_with_uncertainty,
                                                                          data_chunk.energy)

                alpha_temperature, alpha_density = calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
                    dependencies.alpha_temperature_density_calibration_table, alpha_solar_wind_speed,
                    coincidence_count_rates_with_uncertainty,
                    data_chunk.energy, dependencies.efficiency_calibration_table.get_alpha_efficiency_for(epoch))

            except Exception as e:
                logger.info(f"Exception occurred at epoch {epoch}, continuing with fill value", exc_info=True)
            finally:
                epochs.append(epoch)
                alpha_solar_wind_speeds.append(alpha_solar_wind_speed)
                alpha_solar_wind_densities.append(alpha_density)
                alpha_solar_wind_temperatures.append(alpha_temperature)


        alpha_solar_wind_speed_metadata = replace(self.input_metadata, descriptor="alpha-sw")
        alpha_solar_wind_l3_data = SwapiL3AlphaSolarWindData(alpha_solar_wind_speed_metadata, np.array(epochs),
                                                             np.array(alpha_solar_wind_speeds),
                                                             np.array(alpha_solar_wind_temperatures),
                                                             np.array(alpha_solar_wind_densities))
        return alpha_solar_wind_l3_data

    def process_l3a_proton(self, data, dependencies) -> SwapiL3ProtonSolarWindData:
        epochs = []

        proton_solar_wind_speeds = []
        proton_solar_wind_temperatures = []
        proton_solar_wind_density = []
        proton_solar_wind_clock_angles = []
        proton_solar_wind_deflection_angles = []

        for data_chunk in chunk_l2_data(data, 5):
            proton_solar_wind_speed = ufloat(np.nan, np.nan)
            clock_angle = ufloat(np.nan, np.nan)
            deflection_angle = ufloat(np.nan, np.nan)
            proton_density = ufloat(np.nan, np.nan)
            proton_temperature = ufloat(np.nan, np.nan)

            epoch_center_of_chunk = data_chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS
            try:
                if np.any(np.isnan(extract_coarse_sweep(data_chunk.coincidence_count_rate))):
                    raise ValueError("Fill values in input data")
                coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                                  data_chunk.coincidence_count_rate_uncertainty)
                proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                    coincidence_count_rates_with_uncertainty, data_chunk.energy, data_chunk.sci_start_time)

                clock_angle = calculate_clock_angle(dependencies.clock_angle_and_flow_deflection_calibration_table,
                                                    proton_solar_wind_speed, a, phi, b)

                deflection_angle = calculate_deflection_angle(
                    dependencies.clock_angle_and_flow_deflection_calibration_table,
                    proton_solar_wind_speed, a, phi, b)

                proton_temperature, proton_density = calculate_proton_solar_wind_temperature_and_density(
                    dependencies.proton_temperature_density_calibration_table,
                    proton_solar_wind_speed,
                    deflection_angle,
                    clock_angle,
                    coincidence_count_rates_with_uncertainty,
                    data_chunk.energy,
                    dependencies.efficiency_calibration_table.get_proton_efficiency_for(epoch_center_of_chunk)
                )
            except Exception as e:
                epoch = epoch_center_of_chunk
                logger.info(f"Exception occurred at epoch {epoch}, continuing with fill value", exc_info=True)

            proton_solar_wind_speeds.append(proton_solar_wind_speed)
            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)
            proton_solar_wind_density.append(proton_density)
            proton_solar_wind_temperatures.append(proton_temperature)
            epochs.append(epoch_center_of_chunk)

        proton_solar_wind_speed_metadata = replace(self.input_metadata, descriptor="proton-sw")
        proton_solar_wind_l3_data = SwapiL3ProtonSolarWindData(proton_solar_wind_speed_metadata, np.array(epochs),
                                                               np.array(proton_solar_wind_speeds),
                                                               np.array(proton_solar_wind_temperatures),
                                                               np.array(proton_solar_wind_density),
                                                               np.array(proton_solar_wind_clock_angles),
                                                               np.array(proton_solar_wind_deflection_angles))

        return proton_solar_wind_l3_data

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
            center_of_epoch = data_chunk.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS
            instrument_efficiency = dependencies.efficiency_calibration_table.get_proton_efficiency_for(center_of_epoch)
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

        l3b_combined_metadata = self.input_metadata
        l3b_combined_metadata.descriptor = 'combined'
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
