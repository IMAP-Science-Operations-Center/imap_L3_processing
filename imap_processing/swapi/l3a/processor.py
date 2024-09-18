import dataclasses
from dataclasses import dataclass
import numpy as np
from spacepy.pycdf import CDF
from uncertainties import ufloat
from uncertainties.unumpy import uarray

from imap_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS
from imap_processing.models import UpstreamDataDependency
from imap_processing.processor import Processor
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData, SwapiL3AlphaSolarWindData
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_alpha_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable, calculate_deflection_angle, calculate_clock_angle
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_proton_solar_wind_speed
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    TemperatureAndDensityCalibrationTable, calculate_proton_solar_wind_temperature_and_density
from imap_processing.swapi.l3a.utils import read_l2_swapi_data, chunk_l2_data
from imap_processing.utils import download_dependency, upload_data

SWAPI_L2_DESCRIPTOR = "fake-menlo-5-sweeps"
TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR = "density-temperature-lut-text-not-cdf"


@dataclass
class SwapiL3ADependencies:
    data: CDF
    temperature_density_calibration_table: TemperatureAndDensityCalibrationTable
    clock_angle_and_flow_deflection_calibration_table: ClockAngleCalibrationTable

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            data_dependency = next(
                dep for dep in dependencies if dep.descriptor == SWAPI_L2_DESCRIPTOR)
            data_dependency_path = download_dependency(data_dependency)
        except StopIteration:
            raise ValueError(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.")
        except ValueError as e:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{e}")

        try:
            calibration_table_dependency = next(
                dep for dep in dependencies if dep.descriptor == TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
            calibration_table_dependency = dataclasses.replace(calibration_table_dependency, start_date=None,
                                                               end_date=None)
            calibration_table_dependency_path = download_dependency(calibration_table_dependency)
        except StopIteration:
            raise ValueError(f"Missing {TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR} dependency.")
        except ValueError as e:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{e}")

        temperature_and_density_calibration_file = TemperatureAndDensityCalibrationTable.from_file(
            calibration_table_dependency_path)
        data_file = CDF(str(data_dependency_path))

        clock_angle_and_deflection_calibration_table_dependency = UpstreamDataDependency("swapi", "l2", None, None,
                                                                                         "latest",
                                                                                         "clock-angle-and-flow-deflection-lut-text-not-cdf")
        clock_and_deflection_file_path = download_dependency(clock_angle_and_deflection_calibration_table_dependency)

        clock_angle_calibration_file = ClockAngleCalibrationTable.from_file(clock_and_deflection_file_path)
        return cls(data_file, temperature_and_density_calibration_file, clock_angle_calibration_file)


class SwapiL3AProcessor(Processor):

    def process(self):
        dependencies = [
            dataclasses.replace(dep, start_date=self.input_metadata.start_date, end_date=self.input_metadata.end_date)
            for dep in
            self.dependencies]
        dependencies = SwapiL3ADependencies.fetch_dependencies(dependencies)

        data = read_l2_swapi_data(dependencies.data)

        epochs = []

        proton_solar_wind_speeds = []
        proton_solar_wind_temperatures = []
        proton_solar_wind_density = []
        proton_solar_wind_clock_angles = []
        proton_solar_wind_deflection_angles = []

        alpha_solar_wind_speeds = []

        for data_chunk in chunk_l2_data(data, 5):
            coincidence_count_rates_with_uncertainty = uarray(data_chunk.coincidence_count_rate,
                                                              data_chunk.coincidence_count_rate_uncertainty)
            proton_solar_wind_speed, a, phi, b = calculate_proton_solar_wind_speed(
                coincidence_count_rates_with_uncertainty,
                data_chunk.spin_angles, data_chunk.energy, data_chunk.epoch)
            proton_solar_wind_speeds.append(proton_solar_wind_speed)

            temperature, density = calculate_proton_solar_wind_temperature_and_density(
                dependencies.temperature_density_calibration_table,
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

            proton_solar_wind_temperatures.append(temperature)
            proton_solar_wind_density.append(density)
            proton_solar_wind_clock_angles.append(clock_angle)
            proton_solar_wind_deflection_angles.append(deflection_angle)

            epochs.append(data_chunk.epoch[0] + THIRTY_SECONDS_IN_NANOSECONDS)

            alpha_solar_wind_speeds.append(calculate_alpha_solar_wind_speed(
                coincidence_count_rates_with_uncertainty,
                data_chunk.energy
            ))

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
                                                             np.array(alpha_solar_wind_speeds))
        upload_data(alpha_solar_wind_l3_data)
