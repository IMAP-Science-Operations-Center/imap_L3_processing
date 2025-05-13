from dataclasses import dataclass
from typing import Self

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.maps.map_models import RectangularIntensityMapData


@dataclass
class LoL3SpectralFitDependencies:
    map_data: RectangularIntensityMapData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> Self:
        file_names = dependencies.get_file_paths(source="lo")

        if len(file_names) != 1:
            raise ValueError("Incorrect number of dependencies")

        lo_l3_file = file_names[0].name
        cdf = download(lo_l3_file)

        return cls(RectangularIntensityMapData.read_from_path(cdf))
