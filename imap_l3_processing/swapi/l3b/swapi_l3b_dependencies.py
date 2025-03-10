from dataclasses import dataclass

from spacepy.pycdf import CDF

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swapi.descriptors import GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, SWAPI_L2_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_l3_processing.utils import download_dependency


@dataclass
class SwapiL3BDependencies:
    data: CDF
    geometric_factor_calibration_table: GeometricFactorCalibrationTable
    efficiency_calibration_table: EfficiencyCalibrationTable

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]):
        try:
            data_dependency = next(
                dep for dep in dependencies if dep.descriptor == SWAPI_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {SWAPI_L2_DESCRIPTOR} dependency.")

        geometric_factor_calibration_table_dependency = UpstreamDataDependency("swapi", "l2", None, None,
                                                                               "latest",
                                                                               GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR)

        efficiency_calibration_table_dependency = UpstreamDataDependency("swapi", "l2", None, None,
                                                                         "latest",
                                                                         EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR)

        try:
            data_dependency_path = download_dependency(data_dependency)
            downloaded_geometric_factor_file_path = download_dependency(
                geometric_factor_calibration_table_dependency)
            downloaded_efficiency_file_path = download_dependency(
                efficiency_calibration_table_dependency
            )
        except ValueError as e:
            raise ValueError(f"Unexpected files found for SWAPI L3:"
                             f"{e}")

        data_file = CDF(str(data_dependency_path))
        geometric_factor_table = GeometricFactorCalibrationTable.from_file(
            downloaded_geometric_factor_file_path)
        efficiency_table = EfficiencyCalibrationTable(downloaded_efficiency_file_path)

        return cls(data_file, geometric_factor_table, efficiency_table)
