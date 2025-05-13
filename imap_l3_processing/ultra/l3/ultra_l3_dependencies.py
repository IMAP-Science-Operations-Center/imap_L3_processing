from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from imap_data_access import download
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.maps.map_models import HealPixIntensityMapData
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.utils import find_glows_l3e_dependencies


@dataclass
class UltraL3Dependencies:
    ultra_l2_map: HealPixIntensityMapData
    ultra_l1c_pset: list[UltraL1CPSet]
    glows_l3e_sp: list[UltraGlowsL3eData]

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> Self:
        ultra_l2_file_paths = deps.get_file_paths("ultra", "45sensor-pset")
        if len(ultra_l2_file_paths) != 1:
            raise ValueError("Incorrect number of dependencies")
        l2_map_path = download(ultra_l2_file_paths[0])

        hi_l1c_paths = []
        with CDF(str(l2_map_path)) as l2_map:
            map_input_paths = [generate_imap_file_path(file) for file in l2_map.attrs["Parents"]]
            l1c_file_names = [map_input_file.filename.name for map_input_file in map_input_paths if
                              isinstance(map_input_file, ScienceFilePath) and map_input_file.data_level == "l1c"]
            for parent in l1c_file_names:
                hi_l1c_paths.append(download(parent))

        glows_l3e_file_names = find_glows_l3e_dependencies(l1c_file_names, "ultra")
        glows_file_paths = [download(path) for path in glows_l3e_file_names]
        return cls.from_file_paths(l2_map_path, hi_l1c_paths, glows_file_paths)

    @classmethod
    def from_file_paths(cls, l2_map_path: Path, l1c_file_paths: list[Path], glows_file_paths: list[Path]) -> Self:
        ultra_l1c_data = []
        glows_l3e_data = []
        ultra_l2_map = HealPixIntensityMapData.read_from_path(l2_map_path)
        for file_path in l1c_file_paths:
            ultra_l1c_data.append(UltraL1CPSet.read_from_path(file_path))
        for file_path in glows_file_paths:
            glows_l3e_data.append(UltraGlowsL3eData.read_from_path(file_path))
        return cls(ultra_l2_map=ultra_l2_map, ultra_l1c_pset=ultra_l1c_data, glows_l3e_sp=glows_l3e_data)


@dataclass
class UltraL3SpectralFitDependencies:
    ultra_l3_data: HealPixIntensityMapData
    energy_fit_ranges: np.ndarray

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> Self:
        energy_fit_ranges_ancillary_file_path = deps.get_file_paths(source="ultra", descriptor="spx-energy-ranges")
        ultra_map_file_paths = [dep for dep in deps.get_file_paths(source="ultra") if
                                dep != energy_fit_ranges_ancillary_file_path]
        map_file_path = download(ultra_map_file_paths[0].name)
        energy_ranges_file_path = download(energy_fit_ranges_ancillary_file_path[0].name)

        return cls.from_file_paths(map_file_path, energy_ranges_file_path)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, energy_fit_ranges_path: Path):
        map_data = HealPixIntensityMapData.read_from_path(map_file_path)
        energy_fit_ranges = np.loadtxt(energy_fit_ranges_path)
        return cls(map_data, energy_fit_ranges)
