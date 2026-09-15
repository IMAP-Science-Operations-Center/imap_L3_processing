from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_mass_correction_lookup_table import \
    CodiceHiMassCorrectionLookupTable
from imap_l3_processing.codice.l3.hi.models import CodiceL2HiDirectEventData, CodiceL1aHiDirectEvents


@dataclass
class CodiceHiL3aDirectEventsDependencies:
    codice_l2_hi_data: CodiceL2HiDirectEventData
    codice_l1a_hi_data: CodiceL1aHiDirectEvents
    codice_hi_mass_correction_lut_ancillary: CodiceHiMassCorrectionLookupTable

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        l2_hi_direct_events_file_paths = dependencies.get_file_paths("codice", "hi-direct-events", data_type="l2")
        l1a_hi_direct_events_file_paths = dependencies.get_file_paths("codice", "hi-direct-events", data_type="l1a")
        ancillary_file_paths = dependencies.get_file_paths("codice", "l3-hi-mass-correction-lut")

        for download_location_file_path in [*l2_hi_direct_events_file_paths, *l1a_hi_direct_events_file_paths, *ancillary_file_paths]:
            download(download_location_file_path)

        return cls.from_file_paths(l2_hi_direct_events_file_paths[0], l1a_hi_direct_events_file_paths[0], ancillary_file_paths[0])

    @classmethod
    def from_file_paths(cls, codice_l2_hi_cdf: Path, codice_l1a_hi_cdf: Path, codice_hi_mass_correction_lut_ancillary: Path):
        codice_l2_hi_data = CodiceL2HiDirectEventData.read_from_cdf(codice_l2_hi_cdf)
        codice_l1a_hi_data = CodiceL1aHiDirectEvents.read_from_cdf(codice_l1a_hi_cdf)
        codice_hi_mass_correction_lut_ancillary = CodiceHiMassCorrectionLookupTable.from_file(codice_hi_mass_correction_lut_ancillary)

        return cls(codice_l2_hi_data, codice_l1a_hi_data, codice_hi_mass_correction_lut_ancillary)
