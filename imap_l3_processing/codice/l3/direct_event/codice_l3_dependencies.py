from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.direct_event.science.tof_lookup import TOFLookup
from imap_l3_processing.codice.models import CodiceL2HiData


@dataclass
class CodiceL3Dependencies:
    tof_lookup: TOFLookup
    codice_l2_hi_data: CodiceL2HiData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        for dep in dependencies.get_science_inputs():
            if dep.data_type != "l2":
                dependencies.processing_input.remove(dep)

        science_file_paths = dependencies.get_file_paths("codice", "direct-events")
        ancillary_file_paths = dependencies.get_file_paths("codice", "tof-lookup")

        for download_location_file_path in [*science_file_paths, *ancillary_file_paths]:
            download(download_location_file_path)

        return cls.from_file_paths(science_file_paths[0], ancillary_file_paths[0])

    @classmethod
    def from_file_paths(cls, codice_l2_hi_cdf: Path, tof_lookup_path: Path):
        tof_lookup = TOFLookup.read_from_file(tof_lookup_path)
        codice_l2_hi_cdf = CodiceL2HiData.read_from_cdf(codice_l2_hi_cdf)

        return cls(tof_lookup, codice_l2_hi_cdf)
