from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from imap_data_access import download
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection
from imap_processing.ultra.l2.ultra_l2 import ultra_l2
from spacepy.pycdf import CDF

from imap_l3_processing.maps.map_models import HealPixIntensityMapData, RectangularIntensityMapData, \
    SpectralIndexDependencies
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.utils import find_glows_l3e_dependencies


@dataclass
class UltraL3Dependencies:
    ultra_l2_map: HealPixIntensityMapData
    ultra_l1c_pset: list[UltraL1CPSet]
    glows_l3e_sp: list[UltraGlowsL3eData]
    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3Dependencies:
        ultra_l2_file_paths = deps.get_file_paths("ultra")

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
    def from_file_paths(cls, l2_map_path: Path, l1c_file_paths: list[Path], glows_file_paths: list[Path]) -> UltraL3Dependencies:
        ultra_l1c_data = []
        glows_l3e_data = []
        for file_path in l1c_file_paths:
            ultra_l1c_data.append(UltraL1CPSet.read_from_path(file_path))
        for file_path in glows_file_paths:
            glows_l3e_data.append(UltraGlowsL3eData.read_from_path(file_path))
        paths = [l2_map_path] + l1c_file_paths + glows_file_paths
        l1c_paths_dict = {f"l1c_path_{index + 1}": path for index, path in enumerate(l1c_file_paths)}
        l2_healpix_datasets = ultra_l2(l1c_paths_dict)
        l2_healpix_map_data = HealPixIntensityMapData.read_from_xarray(l2_healpix_datasets[0])

        return cls(ultra_l2_map=l2_healpix_map_data, ultra_l1c_pset=ultra_l1c_data, glows_l3e_sp=glows_l3e_data,
                   dependency_file_paths=paths)


@dataclass
class UltraL3SpectralIndexDependencies(SpectralIndexDependencies):
    fit_energy_ranges: np.ndarray

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3SpectralIndexDependencies:
        energy_fit_ranges_ancillary_file_path = deps.get_file_paths(source="ultra", descriptor="spx-energy-ranges")
        ultra_map_file_paths = [dep for dep in deps.get_file_paths(source="ultra") if
                                dep not in energy_fit_ranges_ancillary_file_path]

        if len(ultra_map_file_paths) != 1:
            raise ValueError("Missing Ultra L3 file")

        if len(energy_fit_ranges_ancillary_file_path) != 1:
            raise ValueError("Missing fit energy ranges ancillary file")

        map_file_path = download(ultra_map_file_paths[0].name)
        energy_ranges_file_path = download(energy_fit_ranges_ancillary_file_path[0].name)

        return cls.from_file_paths(map_file_path, energy_ranges_file_path)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, energy_fit_ranges_path: Path):
        map_data = RectangularIntensityMapData.read_from_path(map_file_path)
        energy_fit_ranges = np.loadtxt(energy_fit_ranges_path)
        return cls(map_data, energy_fit_ranges)

    def get_fit_energy_ranges(self) -> np.ndarray:
        return self.fit_energy_ranges
