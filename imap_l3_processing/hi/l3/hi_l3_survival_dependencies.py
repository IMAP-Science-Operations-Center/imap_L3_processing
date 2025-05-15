from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imap_data_access
from imap_data_access import ScienceFilePath
from imap_data_access.file_validation import generate_imap_file_path
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.hi.l3.utils import read_hi_l1c_data, read_glows_l3e_data
from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, SpinPhase
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, HiGlowsL3eData, HiL1cData
from imap_l3_processing.utils import find_glows_l3e_dependencies


@dataclass
class HiL3SurvivalDependencies:
    l2_data: RectangularIntensityMapData
    hi_l1c_data: list[HiL1cData]
    glows_l3e_data: list[HiGlowsL3eData]
    l2_map_descriptor_parts: MapDescriptorParts

    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HiL3SurvivalDependencies:
        l2_map_paths = dependencies.get_file_paths(source="hi")
        assert len(l2_map_paths) == 1
        map_file_path = imap_data_access.download(l2_map_paths[0].name)
        hi_l1c_paths = []
        with CDF(str(map_file_path)) as l2_map:
            map_input_paths = [generate_imap_file_path(file) for file in l2_map.attrs["Parents"]]
            l1c_file_names = [map_input_file.filename.name for map_input_file in map_input_paths if
                              isinstance(map_input_file, ScienceFilePath) and map_input_file.data_level == "l1c"]
            for parent in l1c_file_names:
                hi_l1c_paths.append(imap_data_access.download(parent))
        glows_l3e_file_names = find_glows_l3e_dependencies(l1c_file_names, "hi")
        glows_file_paths = [imap_data_access.download(path) for path in glows_l3e_file_names]
        l2_descriptor = generate_imap_file_path(l2_map_paths[0].name).descriptor
        return cls.from_file_paths(map_file_path, hi_l1c_paths, glows_file_paths, l2_descriptor)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, hi_l1c_paths: list[Path],
                        glows_l3e_paths: list[Path], l2_descriptor: str) -> HiL3SurvivalDependencies:
        glows_l3e_data = list(map(read_glows_l3e_data, glows_l3e_paths))
        l1c_data = list(map(read_hi_l1c_data, hi_l1c_paths))

        paths = [map_file_path] + hi_l1c_paths + glows_l3e_paths
        return cls(l2_data=RectangularIntensityMapData.read_from_path(map_file_path), hi_l1c_data=l1c_data,
                   glows_l3e_data=glows_l3e_data,
                   l2_map_descriptor_parts=parse_map_descriptor(l2_descriptor),
                   dependency_file_paths=paths)


@dataclass
class HiL3SingleSensorFullSpinDependencies:
    ram_dependencies: HiL3SurvivalDependencies
    antiram_dependencies: HiL3SurvivalDependencies

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HiL3SingleSensorFullSpinDependencies:
        parsed_descriptors = [parse_map_descriptor(pi.descriptor) for pi in dependencies.processing_input]

        ram_dependencies = [pi for pi, descriptor in zip(dependencies.processing_input, parsed_descriptors) if
                            descriptor is None or descriptor.spin_phase == SpinPhase.RamOnly]
        antiram_dependencies = [pi for pi, descriptor in zip(dependencies.processing_input, parsed_descriptors) if
                                descriptor is None or descriptor.spin_phase == SpinPhase.AntiRamOnly]

        return cls(
            ram_dependencies=HiL3SurvivalDependencies.fetch_dependencies(ProcessingInputCollection(*ram_dependencies)),
            antiram_dependencies=HiL3SurvivalDependencies.fetch_dependencies(
                ProcessingInputCollection(*antiram_dependencies)))

    @property
    def dependency_file_paths(self) -> list[Path]:
        return self.ram_dependencies.dependency_file_paths + self.antiram_dependencies.dependency_file_paths
