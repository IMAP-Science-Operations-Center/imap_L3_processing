from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoDirectEventData, CodiceLoL1aSWPriorityRates, \
    CodiceLoL1aNSWPriorityRates


@dataclass
class CodiceLoL3a3dDistributionsDependencies:
    l3a_direct_event_data: CodiceLoDirectEventData
    l1a_sw_data: CodiceLoL1aSWPriorityRates
    l1a_nsw_data: CodiceLoL1aNSWPriorityRates
    mass_species_bin_lookup: MassSpeciesBinLookup
    
    @classmethod
    def fetch_dependencies(cls, ):
        pass

    @classmethod
    def from_file_paths(cls, l3a_file_path: Path, l1a_sw_file_path: Path, l1a_nsw_file_path: Path,
                        mass_species_bin_lut: Path):
        return cls(CodiceLoDirectEventData.read_from_cdf(l3a_file_path),
                   CodiceLoL1aSWPriorityRates.read_from_cdf(l1a_sw_file_path),
                   CodiceLoL1aNSWPriorityRates.read_from_cdf(l1a_nsw_file_path),
                   MassSpeciesBinLookup.read_from_csv(mass_species_bin_lut),
                   )
