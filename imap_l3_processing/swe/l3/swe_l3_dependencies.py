from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.models import MagL1dData, UpstreamDataDependency
from imap_l3_processing.swe.l3.models import SweL2Data, SweConfiguration, SwapiL3aProtonData, SweL1bData
from imap_l3_processing.swe.l3.utils import read_l2_swe_data, read_l3a_swapi_proton_data, read_swe_config, \
    read_l1b_swe_data
from imap_l3_processing.utils import read_l1d_mag_data, download_dependency

MAG_L1D_DESCRIPTOR = "norm-mago"
SWAPI_L3A_PROTON_DESCRIPTOR = "proton-sw"
SWE_CONFIG_DESCRIPTOR = "config-json-not-cdf"


@dataclass
class SweL3Dependencies:
    swe_l2_data: SweL2Data
    swe_l1b_data: SweL1bData
    mag_l1d_data: MagL1dData
    swapi_l3a_proton_data: SwapiL3aProtonData
    configuration: SweConfiguration

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]) -> SweL3Dependencies:
        swe_config_dependency = UpstreamDataDependency("swe", "l3", None, None, "latest", SWE_CONFIG_DESCRIPTOR)

        try:
            swe_l2_dependency = next(
                d for d in dependencies if d.instrument == "swe" and d.data_level == "l2")
        except StopIteration:
            raise ValueError(f"Missing SWE dependency.")
        try:
            swe_l1b_dependency = next(
                d for d in dependencies if d.instrument == "swe" and d.data_level == "l1b")
        except StopIteration:
            raise ValueError(f"Missing SWE dependency.")
        try:
            mag_dependency = next(
                d for d in dependencies if d.instrument == "mag"
                and d.descriptor == MAG_L1D_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing MAG {MAG_L1D_DESCRIPTOR} dependency.")
        try:
            swapi_dependency = next(
                d for d in dependencies if d.instrument == "swapi"
                and d.descriptor == SWAPI_L3A_PROTON_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing SWAPI {SWAPI_L3A_PROTON_DESCRIPTOR} dependency.")

        swe_l2_file = download_dependency(swe_l2_dependency)
        swe_l1b_file = download_dependency(swe_l1b_dependency)
        mag_file = download_dependency(mag_dependency)
        swapi_file = download_dependency(swapi_dependency)
        swe_config = download_dependency(swe_config_dependency)

        return cls.from_file_paths(swe_l2_file, swe_l1b_file, mag_file, swapi_file, swe_config)

    @classmethod
    def from_file_paths(cls, swe_l2_file_path: Path, swe_l1b_file_path: Path, mag_file_path: Path,
                        swapi_file_path: Path,
                        configuration_file_path: Path) -> SweL3Dependencies:
        mag_l1d_data = read_l1d_mag_data(mag_file_path)
        swe_l1b_data = read_l1b_swe_data(swe_l1b_file_path)
        swe_l2_data = read_l2_swe_data(swe_l2_file_path)
        swapi_l3a_proton_data = read_l3a_swapi_proton_data(swapi_file_path)
        configuration = read_swe_config(configuration_file_path)

        return cls(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, configuration)
