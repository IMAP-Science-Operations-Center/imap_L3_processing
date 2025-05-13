from dataclasses import dataclass
from pathlib import Path
from typing import Self

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.maps.map_models import RectangularIntensityMapData


@dataclass
class HiL3CombinedMapDependencies:
    maps: list[RectangularIntensityMapData]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> Self:
        input_map_filenames = dependencies.get_file_paths(source="hi")

        map_files = [imap_data_access.download(input_map.name) for input_map in input_map_filenames]
        return cls.from_file_paths(map_files)

    @classmethod
    def from_file_paths(cls, hi_l3_map_paths: list[Path]) -> Self:
        return cls([RectangularIntensityMapData.read_from_path(path) for path in hi_l3_map_paths])
