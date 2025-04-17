from dataclasses import dataclass
from pathlib import Path
from typing import Self

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData


@dataclass
class UltraL3Dependencies:
    ultra_l1c_pset: list[UltraL1CPSet]
    glows_l3e_sp: list[UltraGlowsL3eData]

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> Self:
        ultra_file_paths = deps.get_file_paths("ultra", "45sensor-pset")
        glows_file_paths = deps.get_file_paths("glows", "survival-probabilities-ultra")

        for file_path in [*ultra_file_paths, *glows_file_paths]:
            download(file_path)

        return cls.read_from_file(ultra_file_paths, glows_file_paths)

    @classmethod
    def read_from_file(cls, ultra_file_paths: list[Path], glows_file_paths: list[Path]) -> Self:
        ultra_l1c_data = []
        glows_l3e_data = []
        for file_path in ultra_file_paths:
            ultra_l1c_data.append(UltraL1CPSet.read_from_path(file_path))
        for file_path in glows_file_paths:
            glows_l3e_data.append(UltraGlowsL3eData.read_from_path(file_path))
        return cls(ultra_l1c_pset=ultra_l1c_data, glows_l3e_sp=glows_l3e_data)
