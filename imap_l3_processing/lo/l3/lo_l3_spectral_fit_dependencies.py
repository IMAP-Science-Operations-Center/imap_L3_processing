from dataclasses import dataclass

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.models import RectangularIntensityMapData
from imap_l3_processing.utils import read_rectangular_intensity_map_data_from_cdf


@dataclass
class LoL3SpectralFitDependencies:
    map_data: RectangularIntensityMapData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        file_name = dependencies.get_file_paths(source="lo")[0].name
        cdf = download(file_name)

        return cls(read_rectangular_intensity_map_data_from_cdf(cdf))
