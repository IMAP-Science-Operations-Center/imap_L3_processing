from dataclasses import dataclass

from spacepy.pycdf import CDF

from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    ProtonTemperatureAndDensityCalibrationTable
from imap_processing.utils import download_dependency

SWAPI_L2_DESCRIPTOR = "sci"
PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR = "density-temperature-lut-text-not-cdf"
ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR = "alpha-density-temperature-lut-text-not-cdf"
CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR = "clock-angle-and-flow-deflection-lut-text-not-cdf"
GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR = "energy-gf-lut-not-cdf"


@dataclass
class SwapiL3ADependencies:
    data: CDF
    proton_temperature_density_calibration_table: ProtonTemperatureAndDensityCalibrationTable
    alpha_temperature_density_calibration_table: AlphaTemperatureDensityCalibrationTable
    clock_angle_and_flow_deflection_calibration_table: ClockAngleCalibrationTable

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            data_dependency = next(
                dep for dep in dependencies if dep.descriptor == SWAPI_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.")

        proton_density_and_temperature_calibration_file = UpstreamDataDependency("swapi", "l2", None, None,
                                                                                 "latest",
                                                                                 PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
        alpha_density_and_temperature_calibration_file = UpstreamDataDependency("swapi", "l2", None, None,
                                                                                "latest",
                                                                                ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
        clock_angle_and_deflection_calibration_table_dependency = UpstreamDataDependency("swapi", "l2", None, None,
                                                                                         "latest",
                                                                                         CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR)

        try:
            data_dependency_path = download_dependency(data_dependency)
            proton_density_and_temperature_calibration_file_path = download_dependency(
                proton_density_and_temperature_calibration_file)
            alpha_density_and_temperature_calibration_file_path = download_dependency(
                alpha_density_and_temperature_calibration_file)
            clock_and_deflection_file_path = download_dependency(
                clock_angle_and_deflection_calibration_table_dependency)
        except ValueError as e:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{e}")

        data_file = CDF(str(data_dependency_path))
        proton_temperature_and_density_calibration_file = ProtonTemperatureAndDensityCalibrationTable.from_file(
            proton_density_and_temperature_calibration_file_path)
        alpha_density_and_temperature_calibration_file = AlphaTemperatureDensityCalibrationTable.from_file(
            alpha_density_and_temperature_calibration_file_path)
        clock_angle_calibration_file = ClockAngleCalibrationTable.from_file(clock_and_deflection_file_path)

        return cls(data_file, proton_temperature_and_density_calibration_file,
                   alpha_density_and_temperature_calibration_file, clock_angle_calibration_file)
