from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.swapi.descriptors import EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, \
    GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR, SWAPI_L2_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable


@dataclass
class SwapiL3BDependencies:
    data: SwapiL2Data
    geometric_factor_calibration_table: GeometricFactorCalibrationTable
    efficiency_calibration_table: EfficiencyCalibrationTable

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        science_dependency_file = dependencies.get_file_paths(source='swapi', descriptor=SWAPI_L2_DESCRIPTOR)
        geometric_factor_lookup_table_file = dependencies.get_file_paths(source='swapi',
                                                                         descriptor=GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR)
        efficiency_table_lookup_file = dependencies.get_file_paths(source='swapi',
                                                                   descriptor=EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR)

        download(science_dependency_file[0])
        download(geometric_factor_lookup_table_file[0])
        download(efficiency_table_lookup_file[0])

        return cls.from_file_paths(science_dependency_file[0],
                                   geometric_factor_lookup_table_file[0],
                                   efficiency_table_lookup_file[0])

    @classmethod
    def from_file_paths(cls, science_dependency_path: Path, geometric_factor_calibration_path: Path,
                        efficiency_calibration_table_path: Path):
        swapi_l2_data = read_l2_swapi_data(CDF(str(science_dependency_path)))
        geometric_factor_calibration_lookup = GeometricFactorCalibrationTable.from_file(
            geometric_factor_calibration_path)
        efficiency_calibration_table = EfficiencyCalibrationTable(efficiency_calibration_table_path)

        return cls(swapi_l2_data,
                   geometric_factor_calibration_lookup,
                   efficiency_calibration_table)
