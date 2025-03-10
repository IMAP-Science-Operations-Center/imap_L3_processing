from dataclasses import dataclass
from pathlib import Path

from spacepy.pycdf import CDF

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swapi.descriptors import SWAPI_L2_DESCRIPTOR, PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, \
    ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR, \
    GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, INSTRUMENT_RESPONSE_LOOKUP_TABLE, DENSITY_OF_NEUTRAL_HELIUM
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_temperature_and_density import \
    AlphaTemperatureDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_clock_and_deflection_angles import \
    ClockAngleCalibrationTable
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_temperature_and_density import \
    ProtonTemperatureAndDensityCalibrationTable
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTableCollection
from imap_l3_processing.utils import download_dependency


@dataclass
class SwapiL3ADependencies:
    data: CDF
    proton_temperature_density_calibration_table: ProtonTemperatureAndDensityCalibrationTable
    alpha_temperature_density_calibration_table: AlphaTemperatureDensityCalibrationTable
    clock_angle_and_flow_deflection_calibration_table: ClockAngleCalibrationTable
    geometric_factor_calibration_table: GeometricFactorCalibrationTable
    instrument_response_calibration_table: InstrumentResponseLookupTableCollection
    density_of_neutral_helium_calibration_table: DensityOfNeutralHeliumLookupTable

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            data_dependency = next(
                dep for dep in dependencies if dep.descriptor == SWAPI_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.")
        try:
            data_dependency_path = download_dependency(data_dependency)

            proton_density_and_temperature_calibration_file_path = cls.get_lookup_table_with_descriptor(
                PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
            alpha_density_and_temperature_calibration_file_path = cls.get_lookup_table_with_descriptor(
                ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR)
            clock_and_deflection_file_path = cls.get_lookup_table_with_descriptor(
                CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR)
            geometric_factor_calibration_table_path = cls.get_lookup_table_with_descriptor(
                GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR)
            instrument_response_table_path = cls.get_lookup_table_with_descriptor(INSTRUMENT_RESPONSE_LOOKUP_TABLE)
            neutral_helium_table_path = cls.get_lookup_table_with_descriptor(DENSITY_OF_NEUTRAL_HELIUM)
        except ValueError as e:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{e}")

        data_file = CDF(str(data_dependency_path))
        proton_temperature_and_density_calibration_table = ProtonTemperatureAndDensityCalibrationTable.from_file(
            proton_density_and_temperature_calibration_file_path)
        alpha_density_and_temperature_calibration_table = AlphaTemperatureDensityCalibrationTable.from_file(
            alpha_density_and_temperature_calibration_file_path)
        clock_angle_calibration_table = ClockAngleCalibrationTable.from_file(clock_and_deflection_file_path)
        geometric_factor_table = GeometricFactorCalibrationTable.from_file(geometric_factor_calibration_table_path)
        instrument_response_table = InstrumentResponseLookupTableCollection.from_file(instrument_response_table_path)
        helium_table = DensityOfNeutralHeliumLookupTable.from_file(neutral_helium_table_path)

        return cls(data_file, proton_temperature_and_density_calibration_table,
                   alpha_density_and_temperature_calibration_table, clock_angle_calibration_table,
                   geometric_factor_table, instrument_response_table, helium_table)

    @staticmethod
    def get_lookup_table_with_descriptor(descriptor: str) -> Path:
        dependency = UpstreamDataDependency("swapi", "l2", None, None, "latest", descriptor)
        return download_dependency(dependency)
