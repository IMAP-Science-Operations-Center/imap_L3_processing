from dataclasses import dataclass

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.maps.map_models import ISNRateData


@dataclass
class LoL3ISNBackgroundSubtractedDependencies:
    map_data: ISNRateData
    
    @classmethod
    def fetch_dependencies(cls, processing_input_collection: ProcessingInputCollection):
        file_names = processing_input_collection.get_file_paths(source="lo")

        lo_l2_file = file_names[0].name

        cdf = download(lo_l2_file)

        return cls(ISNRateData.read_from_path(cdf))
