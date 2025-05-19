from dataclasses import dataclass
from pathlib import Path
from typing import Self

from imap_data_access import ProcessingInputCollection, download

from imap_l3_processing.maps.map_models import HealPixIntensityMapData


@dataclass
class UltraL3ToRectangularDependencies:
    healpix_map_data: HealPixIntensityMapData

    @classmethod
    def fetch_dependencies(cls, deps: ProcessingInputCollection) -> Self:
        [l3_ultra_map] = deps.get_file_paths("ultra", None, "l3")
        l3_map_path = download(l3_ultra_map.name)
        return cls.from_file_maps(l3_map_path)

    @classmethod
    def from_file_maps(cls, healpix_map_path: Path):
        return cls(HealPixIntensityMapData.read_from_path(healpix_map_path))
