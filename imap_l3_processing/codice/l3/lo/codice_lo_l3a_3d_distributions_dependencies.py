from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_direct_events_dependencies import DIRECT_EVENTS_DESCRIPTOR, \
    SW_PRIORITY_DESCRIPTOR, NSW_PRIORITY_DESCRIPTOR
from imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup import GeometricFactorLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoDirectEventData, CodiceLoL1aSWPriorityRates, \
    CodiceLoL1aNSWPriorityRates

MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR = "mass-species-bin-lookup"
GEOMETRIC_FACTOR_LOOKUP_DESCRIPTOR = "lo-geometric-factors"


@dataclass
class CodiceLoL3a3dDistributionsDependencies:
    l3a_direct_event_data: CodiceLoDirectEventData
    l1a_sw_data: CodiceLoL1aSWPriorityRates
    l1a_nsw_data: CodiceLoL1aNSWPriorityRates
    mass_species_bin_lookup: MassSpeciesBinLookup
    geometric_factors_lookup: GeometricFactorLookup

    @classmethod
    def fetch_dependencies(cls, processing_input_collection: ProcessingInputCollection):
        l3a_direct_event_paths = []
        l1a_sw_paths = []
        l1a_nsw_paths = []
        for input in processing_input_collection.get_science_inputs():
            match input:
                case ScienceInput(data_type="l3a", source="codice", descriptor=descriptor) \
                    if descriptor == DIRECT_EVENTS_DESCRIPTOR:
                    l3a_direct_event_paths.extend(input.filename_list)
                case ScienceInput(data_type="l1a", source="codice", descriptor=descriptor) \
                    if descriptor == SW_PRIORITY_DESCRIPTOR:
                    l1a_sw_paths.extend(input.filename_list)
                case ScienceInput(data_type="l1a", source="codice", descriptor=descriptor) \
                    if descriptor == NSW_PRIORITY_DESCRIPTOR:
                    l1a_nsw_paths.extend(input.filename_list)

        [direct_events_path] = l3a_direct_event_paths
        [sw_priority_rates_path] = l1a_sw_paths
        [nsw_priority_rates_path] = l1a_nsw_paths
        [mass_species_bin_path] = processing_input_collection.get_file_paths(source='codice',
                                                                             descriptor=MASS_SPECIES_BIN_LOOKUP_DESCRIPTOR)
        [geometric_factor_path] = processing_input_collection.get_file_paths(source='codice',
                                                                             descriptor=GEOMETRIC_FACTOR_LOOKUP_DESCRIPTOR)

        direct_events_downloaded_path = imap_data_access.download(direct_events_path)
        sw_priority_rates_downloaded_path = imap_data_access.download(sw_priority_rates_path)
        nsw_priority_rates_downloaded_path = imap_data_access.download(nsw_priority_rates_path)
        mass_species_bin_downloaded_path = imap_data_access.download(mass_species_bin_path.name)
        geometric_factor_downloaded_path = imap_data_access.download(geometric_factor_path.name)

        return cls.from_file_paths(l3a_file_path=direct_events_downloaded_path,
                                   l1a_sw_file_path=sw_priority_rates_downloaded_path,
                                   l1a_nsw_file_path=nsw_priority_rates_downloaded_path,
                                   mass_species_bin_lut=mass_species_bin_downloaded_path,
                                   geometric_factors_lut=geometric_factor_downloaded_path, )

    @classmethod
    def from_file_paths(cls, l3a_file_path: Path, l1a_sw_file_path: Path, l1a_nsw_file_path: Path,
                        mass_species_bin_lut: Path, geometric_factors_lut: Path):
        return cls(CodiceLoDirectEventData.read_from_cdf(l3a_file_path),
                   CodiceLoL1aSWPriorityRates.read_from_cdf(l1a_sw_file_path),
                   CodiceLoL1aNSWPriorityRates.read_from_cdf(l1a_nsw_file_path),
                   MassSpeciesBinLookup.read_from_csv(mass_species_bin_lut),
                   GeometricFactorLookup.read_from_csv(geometric_factors_lut)
                   )
