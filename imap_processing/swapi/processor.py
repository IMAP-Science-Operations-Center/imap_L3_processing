import dataclasses

import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.processor import Processor
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, SwapiL3AlphaSolarWindData
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed, \
    calculate_combined_sweeps
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    calculate_deflection_angle, calculate_clock_angle
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    calculate_proton_solar_wind_temperature_and_density
from imap_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_processing.swapi.l3a.utils import read_l2_swapi_data, chunk_l2_data
from imap_processing.swapi.l3b.models import SwapiL3BCombinedVDF
from imap_processing.swapi.l3b.science.calculate_solar_wind_vdf import calculate_proton_solar_wind_vdf, \
    calculate_alpha_solar_wind_vdf
from imap_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_processing.swapi.parameters import INSTRUMENT_EFFICIENCY
from imap_processing.utils import upload_data


class SwapiProcessor(Processor):

    def process(self):
        dependencies = [
            dataclasses.replace(dep, start_date=self.input_metadata.start_date, end_date=self.input_metadata.end_date)
            for dep in
            self.dependencies]

        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = SwapiL3ADependencies.fetch_dependencies(dependencies)
            data = read_l2_swapi_data(l3a_dependencies.data)
            self.process_l3a(data, l3a_dependencies)
        elif self.input_metadata.data_level == "l3b":
            l3b_dependencies = SwapiL3BDependencies.fetch_dependencies(dependencies)
            data = read_l2_swapi_data(l3b_dependencies.data)
            self.process_l3b(data, l3b_dependencies)

    def process_l3a(self, data, dependencies):
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
                coincidence_count_rates_with_uncertainty,
                data_chunk.spin_angles, data_chunk.energy, data_chunk.epoch)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)

            proton_temperature, proton_density = calculate_proton_solar_wind_temperature_and_density(
                dependencies.proton_temperature_density_calibration_table,
                proton_solar_wind_speed,
                ufloat(0.01, 1.0),
                phi,
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy)

            clock_angle = calculate_clock_angle(dependencies.clock_angle_and_flow_deflection_calibration_table,
                                                proton_solar_wind_speed, a, phi, b)

            deflection_angle = calculate_deflection_angle(
                dependencies.clock_angle_and_flow_deflection_calibration_table,
                proton_solar_wind_speed, a, phi, b)

            proton_solar_wind_temperatures.append(proton_temperature)
            proton_solar_wind_density.append(proton_density)
            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)

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

        proton_solar_wind_speed_metadata = self.input_metadata.to_upstream_data_dependency("proton-sw")

        proton_solar_wind_l3_data = SwapiL3ProtonSolarWindData(proton_solar_wind_speed_metadata, np.array(epochs),
                                                               np.array(proton_solar_wind_speeds),
                                                               np.array(proton_solar_wind_temperatures),
                                                               np.array(proton_solar_wind_density),
                                                               np.array(proton_solar_wind_clock_angles),
                                                               np.array(proton_solar_wind_deflection_angles))
        upload_data(proton_solar_wind_l3_data)

        alpha_solar_wind_speed_metadata = self.input_metadata.to_upstream_data_dependency("alpha-sw")
        alpha_solar_wind_l3_data = SwapiL3AlphaSolarWindData(alpha_solar_wind_speed_metadata, np.array(epochs),
                                                             np.array(alpha_solar_wind_speeds),
                                                             np.array(alpha_solar_wind_temperatures),
                                                             np.array(alpha_solar_wind_densities))
        upload_data(alpha_solar_wind_l3_data)

    def process_l3b(self, data, dependencies):
        epochs = []
        cdf_proton_velocities = []
        cdf_proton_probabilities = []
        cdf_alpha_velocities = []
        cdf_alpha_probabilities = []

        for data_chunk in chunk_l2_data(data, 50):
            coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                              data_chunk.coincidence_count_rate_uncertainty)
            average_coincident_count_rates, energies = calculate_combined_sweeps(
                coincidence_count_rates_with_uncertainty, data_chunk.energy)

            proton_velocities, proton_probabilities = calculate_proton_solar_wind_vdf(energies,
                                                                                      average_coincident_count_rates,
                                                                                      INSTRUMENT_EFFICIENCY,
                                                                                      dependencies.geometric_factor_calibration_table)

            alpha_velocities, alpha_probabilities = calculate_alpha_solar_wind_vdf(energies,
                                                                                   average_coincident_count_rates,
                                                                                   INSTRUMENT_EFFICIENCY,
                                                                                   dependencies.geometric_factor_calibration_table)

            epochs.append(data_chunk.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS)
            cdf_proton_velocities.append(proton_velocities)
            cdf_proton_probabilities.append(proton_probabilities)

            cdf_alpha_velocities.append(alpha_velocities)
            cdf_alpha_probabilities.append(alpha_probabilities)

        l3b_combined_metadata = self.input_metadata.to_upstream_data_dependency("combined")
        l3b_combined_vdf = SwapiL3BCombinedVDF(l3b_combined_metadata, epochs, np.array(cdf_proton_velocities),
                                               np.array(cdf_proton_probabilities), np.array(cdf_alpha_velocities),
                                               np.array(cdf_alpha_probabilities))

        upload_data(l3b_combined_vdf)
