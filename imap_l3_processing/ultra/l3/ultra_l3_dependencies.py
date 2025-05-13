from dataclasses import dataclass
from pathlib import Path
from typing import Self

from imap_data_access import download
from imap_data_access.file_validation import generate_imap_file_path, ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.map_models import HealPixIntensityMapData
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
