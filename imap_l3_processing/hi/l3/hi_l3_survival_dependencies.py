from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_data_access import ScienceFilePath
from imap_data_access.file_validation import generate_imap_file_path
from spacepy.pycdf import CDF

from imap_l3_processing.hi.l3.models import HiMapData, HiL1cData, GlowsL3eData, HiIntensityMapData
from imap_l3_processing.hi.l3.utils import read_hi_l2_data, read_hi_l1c_data, read_glows_l3e_data, MapDescriptorParts, \
    parse_map_descriptor, SpinPhase
from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.utils import download_dependency, download_dependency_from_path, find_glows_l3e_dependencies


@dataclass
class HiL3SurvivalDependencies:
    l2_data: HiIntensityMapData
    hi_l1c_data: list[HiL1cData]
    glows_l3e_data: list[GlowsL3eData]
    l2_map_descriptor_parts: MapDescriptorParts

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]) -> HiL3SurvivalDependencies:
        upstream_map_dependency = next(dep for dep in dependencies if dep.data_level == "l2")
        map_file_path = download_dependency(upstream_map_dependency)
        hi_l1c_paths = []
        with CDF(str(map_file_path)) as l2_map:
            map_input_paths = [generate_imap_file_path(file) for file in l2_map.attrs["Parents"]]
            l1c_file_names = [map_input_file.filename.name for map_input_file in map_input_paths if
                              isinstance(map_input_file, ScienceFilePath) and map_input_file.data_level == "l1c"]
            for parent in l1c_file_names:
                hi_l1c_paths.append(download_dependency_from_path(parent))

        glows_l3e_file_names = find_glows_l3e_dependencies(l1c_file_names, "hi")
        glows_file_paths = [download_dependency_from_path(path) for path in glows_l3e_file_names]
        return cls.from_file_paths(map_file_path, hi_l1c_paths, glows_file_paths, upstream_map_dependency.descriptor)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, hi_l1c_paths: list[Path],
                        glows_l3e_paths: list[Path], l2_descriptor: str) -> HiL3SurvivalDependencies:
        glows_l3e_data = list(map(read_glows_l3e_data, glows_l3e_paths))
        l1c_data = list(map(read_hi_l1c_data, hi_l1c_paths))

        return cls(l2_data=read_hi_l2_data(map_file_path), hi_l1c_data=l1c_data, glows_l3e_data=glows_l3e_data,
                   l2_map_descriptor_parts=parse_map_descriptor(l2_descriptor))


@dataclass
class HiL3SingleSensorFullSpinDependencies:
    ram_dependencies: HiL3SurvivalDependencies
    antiram_dependencies: HiL3SurvivalDependencies

    @classmethod
    def fetch_dependencies(cls, dependencies: list[UpstreamDataDependency]) -> HiL3SingleSensorFullSpinDependencies:
        ram_dependencies = []
        antiram_dependencies = []

        for dep in dependencies:
            map_descriptor_parts = parse_map_descriptor(dep.descriptor)
            if map_descriptor_parts is not None:
                match map_descriptor_parts.spin_phase:
                    case SpinPhase.RamOnly:
                        ram_dependencies.append(dep)
                    case SpinPhase.AntiRamOnly:
                        antiram_dependencies.append(dep)

        assert len(ram_dependencies) == 1 and len(antiram_dependencies) == 1

        return cls(ram_dependencies=HiL3SurvivalDependencies.fetch_dependencies(ram_dependencies),
                   antiram_dependencies=HiL3SurvivalDependencies.fetch_dependencies(antiram_dependencies))
