from dataclasses import dataclass
from pathlib import Path
from typing import Self

import imap_data_access
from imap_data_access import ProcessingInputCollection

from imap_l3_processing.maps.map_models import RectangularIntensityMapData

@dataclass
class LoCombinedDependencies:
    map_data: list[RectangularIntensityMapData]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> Self:
        lo_inputs = dependencies.get_file_paths(source="lo")
        downloaded_paths = [imap_data_access.download(Path(i).name) for i in lo_inputs]

        return cls.from_file_paths(downloaded_paths)

    @classmethod
    def from_file_paths(cls, file_paths: list[Path]) -> Self:
        return cls(map_data=[RectangularIntensityMapData.read_from_path(p) for p in file_paths])

