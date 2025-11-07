from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import imap_data_access
import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from imap_l3_processing.utils import get_dependency_paths_by_descriptor
from imap_processing.ultra.l2.ultra_l2 import ultra_l2

from imap_l3_processing.maps.map_models import HealPixIntensityMapData, RectangularIntensityMapData, \
    SpectralIndexDependencies
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData


@dataclass
class UltraL3Dependencies:
    ultra_l2_map: HealPixIntensityMapData
    ultra_l1c_pset: list[UltraL1CPSet]
    glows_l3e_sp: list[UltraGlowsL3eData]
    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3Dependencies:
        ultra_l2_names = deps.get_file_paths("ultra", data_type="l2")
        assert len(ultra_l2_names) == 1, f"Incorrect number of map dependencies: {len(ultra_l2_names)}"
        ultra_l2_name = ultra_l2_names[0]

        ultra_l1c_names = deps.get_file_paths("ultra", data_type="l1c")
        glows_l3e_names = deps.get_file_paths("glows")

        l2_map_path = imap_data_access.download(ultra_l2_name)
        ultra_l1c_downloaded_paths = [imap_data_access.download(l1c) for l1c in ultra_l1c_names]
        glows_l3e_download_paths = [imap_data_access.download(path) for path in glows_l3e_names]

        return cls.from_file_paths(l2_map_path, ultra_l1c_downloaded_paths, glows_l3e_download_paths)

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

        map_file_path = imap_data_access.download(ultra_map_file_paths[0].name)
        energy_ranges_file_path = imap_data_access.download(energy_fit_ranges_ancillary_file_path[0].name)

        return cls.from_file_paths(map_file_path, energy_ranges_file_path)

    @classmethod
    def from_file_paths(cls, map_file_path: Path, energy_fit_ranges_path: Path):
        map_data = RectangularIntensityMapData.read_from_path(map_file_path)
        energy_fit_ranges = np.loadtxt(energy_fit_ranges_path)
        return cls(map_data, energy_fit_ranges)

    def get_fit_energy_ranges(self) -> np.ndarray:
        return self.fit_energy_ranges

@dataclass
class UltraL3CombinedDependencies:
    u45_l2_map: HealPixIntensityMapData
    u90_l2_map: HealPixIntensityMapData
    u45_l1c_psets: list[UltraL1CPSet]
    u90_l1c_psets: list[UltraL1CPSet]
    glows_l3e_psets: list[UltraGlowsL3eData]
    dependency_file_paths: list[Path] = field(default_factory=list)

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> UltraL3CombinedDependencies:
        descriptors = ["u45", "u90", "45sensor-spacecraftpset", "90sensor-spacecraftpset", "survival-probability-ul"]
        file_paths = get_dependency_paths_by_descriptor(deps=deps, descriptors=descriptors)

        assert len(file_paths['u45']) == 1
        assert len(file_paths['u90']) == 1

        u45_pset_paths = [imap_data_access.download(pset) for pset in file_paths['45sensor-spacecraftpset']]
        u90_pset_paths = [imap_data_access.download(pset) for pset in file_paths['90sensor-spacecraftpset']]
        glows_l3e_pset_paths = [imap_data_access.download(pset) for pset in file_paths['survival-probability-ul']]
        u45_map_path = imap_data_access.download(file_paths['u45'][0])
        u90_map_path = imap_data_access.download(file_paths['u90'][0])

        return cls.from_file_paths(u45_pset_paths, u90_pset_paths, glows_l3e_pset_paths, u45_map_path, u90_map_path)

    @classmethod
    def from_file_paths(cls, u45_pset_paths: list[Path], u90_pset_paths: list[Path], glows_l3e_pset_paths: list[Path], u45_map_path: Path, u90_map_path: Path) -> UltraL3CombinedDependencies:
        u45_l1c_psets = []
        u90_l1c_psets = []
        survival_probability_ul_pset = []

        for pset in u45_pset_paths:
            u45_l1c_psets.append(UltraL1CPSet.read_from_path(pset))

        for pset in u90_pset_paths:
            u90_l1c_psets.append(UltraL1CPSet.read_from_path(pset))

        for pset in glows_l3e_pset_paths:
            survival_probability_ul_pset.append(UltraGlowsL3eData.read_from_path(pset))

        l1c_u45_paths_dict = {f"l1c_path_{index + 1}": path for index, path in enumerate(u45_pset_paths)}
        l2_u45_maps = ultra_l2(l1c_u45_paths_dict)
        l2_u45_healpix_map_data = HealPixIntensityMapData.read_from_xarray(l2_u45_maps[0])

        l1c_u90_paths_dict = {f"l1c_path_{index + 1}": path for index, path in enumerate(u90_pset_paths)}
        l2_u90_maps = ultra_l2(l1c_u90_paths_dict)
        l2_u90_healpix_map_data = HealPixIntensityMapData.read_from_xarray(l2_u90_maps[0])

        return cls(u45_l2_map=l2_u45_healpix_map_data, u90_l2_map=l2_u90_healpix_map_data,
                   u45_l1c_psets=u45_l1c_psets, u90_l1c_psets=u90_l1c_psets, glows_l3e_psets=survival_probability_ul_pset,
                   dependency_file_paths=[*u45_pset_paths, *u90_pset_paths, *glows_l3e_pset_paths, u45_map_path, u90_map_path])
