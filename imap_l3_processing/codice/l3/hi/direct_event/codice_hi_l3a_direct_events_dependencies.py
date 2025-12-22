from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.models import CodiceL2HiDirectEventData


@dataclass
class CodiceHiL3aDirectEventsDependencies:
    codice_l2_hi_data: CodiceL2HiDirectEventData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        for dep in dependencies.get_science_inputs():
            if dep.data_type != "l2":
                dependencies.processing_input.remove(dep)

        science_file_paths = dependencies.get_file_paths("codice", "hi-direct-events")

        for download_location_file_path in science_file_paths:
            download(download_location_file_path)

        return cls.from_file_paths(science_file_paths[0])

    @classmethod
    def from_file_paths(cls, codice_l2_hi_cdf: Path):
        codice_l2_hi_cdf = CodiceL2HiDirectEventData.read_from_cdf(codice_l2_hi_cdf)

        return cls(codice_l2_hi_cdf)
