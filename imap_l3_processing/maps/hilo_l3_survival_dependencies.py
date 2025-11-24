from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imap_data_access
from imap_data_access import ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.hi.l3.utils import read_l1c_rectangular_pointing_set_data, read_glows_l3e_data
from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, SpinPhase
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, GlowsL3eRectangularMapInputData, \
    InputRectangularPointingSet
from imap_l3_processing.models import Instrument


@dataclass
class HiLoL3SurvivalDependencies:
    l2_data: RectangularIntensityMapData
    l1c_data: list[InputRectangularPointingSet]
    glows_l3e_data: list[GlowsL3eRectangularMapInputData]
    l2_map_descriptor_parts: MapDescriptorParts

    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection,
                           instrument: Instrument) -> HiLoL3SurvivalDependencies:
        [l2_map_path] = dependencies.get_file_paths(source=instrument.value, data_type="l2")
        l1c_paths = dependencies.get_file_paths(source=instrument.value, data_type="l1c")
        glows_paths = dependencies.get_file_paths(source="glows")

        map_file_path = imap_data_access.download(l2_map_path.name)
        l1c_downloaded_paths = [imap_data_access.download(l1c.name) for l1c in l1c_paths]
        glows_downloaded_paths = [imap_data_access.download(path.name) for path in glows_paths]

        l2_descriptor = ScienceFilePath(l2_map_path).descriptor

        return cls.from_file_paths(map_file_path, l1c_downloaded_paths, glows_downloaded_paths, l2_descriptor)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, l1c_paths: list[Path],
                        glows_l3e_paths: list[Path], l2_descriptor: str) -> HiLoL3SurvivalDependencies:
        glows_l3e_data = list(map(read_glows_l3e_data, glows_l3e_paths))
        l1c_data = list(map(read_l1c_rectangular_pointing_set_data, l1c_paths))

        l2_data = RectangularIntensityMapData.read_from_path(map_file_path)

        paths = [map_file_path] + l1c_paths + glows_l3e_paths

        return cls(l2_data, l1c_data=l1c_data,
                   glows_l3e_data=glows_l3e_data,
                   l2_map_descriptor_parts=parse_map_descriptor(l2_descriptor),
                   dependency_file_paths=paths)


@dataclass
class HiL3SingleSensorFullSpinDependencies:
    ram_dependencies: HiLoL3SurvivalDependencies
    antiram_dependencies: HiLoL3SurvivalDependencies

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HiL3SingleSensorFullSpinDependencies:
        parsed_descriptors = [parse_map_descriptor(pi.descriptor) for pi in dependencies.processing_input]

        ram_dependencies = [pi for pi, descriptor in zip(dependencies.processing_input, parsed_descriptors) if
                            descriptor is None or descriptor.spin_phase == SpinPhase.RamOnly]

        antiram_dependencies = [pi for pi, descriptor in zip(dependencies.processing_input, parsed_descriptors) if
                                descriptor is None or descriptor.spin_phase == SpinPhase.AntiRamOnly]

        return cls(
            ram_dependencies=HiLoL3SurvivalDependencies.fetch_dependencies(
                ProcessingInputCollection(*ram_dependencies), Instrument.IMAP_HI),
            antiram_dependencies=HiLoL3SurvivalDependencies.fetch_dependencies(
                ProcessingInputCollection(*antiram_dependencies), Instrument.IMAP_HI))

    @property
    def dependency_file_paths(self) -> list[Path]:
        return self.ram_dependencies.dependency_file_paths + self.antiram_dependencies.dependency_file_paths
