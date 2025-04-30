from dataclasses import dataclass
from pathlib import Path

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData, \
    CodiceLoL2DirectEventData, CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.utils import download_dependency_from_path

SW_SPECIES_DESCRIPTOR = 'sw-species'
MASS_PER_CHARGE_DESCRIPTOR = 'mass-per-charge'
DIRECT_EVENTS_DESCRIPTOR = 'direct-events'
PRIORITY_RATES_DESCRIPTOR = 'priority-rates'
MASS_COEFFICIENT_DESCRIPTOR = 'mass-coefficient-lookup'
SW_PRIORITY_DESCRIPTOR = 'lo-sw-priority'
NSW_PRIORITY_DESCRIPTOR = 'lo-nsw-priority'


@dataclass
class CodiceLoL3aDependencies:
    codice_lo_l1a_sw_priority_rates: CodiceLoL1aSWPriorityRates
    codice_lo_l1a_nsw_priority_rates: CodiceLoL1aNSWPriorityRates
    codice_l2_lo_data: CodiceLoL2SWSpeciesData
    codice_l2_direct_events: CodiceLoL2DirectEventData
    mass_per_charge_lookup: MassPerChargeLookup
    mass_coefficient_lookup: MassCoefficientLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        for dep in dependencies.get_science_inputs():
            if dep.data_type[
               :2] != 'l2' and dep.descriptor != SW_PRIORITY_DESCRIPTOR and dep.descriptor != NSW_PRIORITY_DESCRIPTOR:
                dependencies.processing_input.remove(dep)

        sw_priority_rates_paths = dependencies.get_file_paths(source='codice', descriptor=SW_PRIORITY_DESCRIPTOR)
        nsw_priority_rates_paths = dependencies.get_file_paths(source='codice', descriptor=NSW_PRIORITY_DESCRIPTOR)
        sectored_intensities_file_paths = dependencies.get_file_paths(source='codice',
                                                                      descriptor=SW_SPECIES_DESCRIPTOR)
        direct_events_path = dependencies.get_file_paths(source='codice', descriptor=DIRECT_EVENTS_DESCRIPTOR)
        mass_per_charge_ancillary_file_path = dependencies.get_file_paths(source='codice',
                                                                          descriptor=MASS_PER_CHARGE_DESCRIPTOR)
        mass_coefficients_file_path = dependencies.get_file_paths(source='codice',
                                                                  descriptor=MASS_COEFFICIENT_DESCRIPTOR)

        for file_path in [*sw_priority_rates_paths, *nsw_priority_rates_paths, *sectored_intensities_file_paths,
                          *direct_events_path,
                          *mass_per_charge_ancillary_file_path, *mass_coefficients_file_path]:
            download_dependency_from_path(file_path)

        return cls.from_file_paths(
            sw_priority_rates_paths[0],
            nsw_priority_rates_paths[0],
            sectored_intensities_file_paths[0],
            direct_events_path[0],
            mass_per_charge_ancillary_file_path[0],
            mass_coefficients_file_path[0])

    @classmethod
    def from_file_paths(cls, sw_priority_rates_cdf: Path, nsw_priority_rates_cdf: Path, codice_l2_lo_cdf: Path,
                        direct_event_path: Path,
                        mass_per_charge_lookup_path: Path, mass_coefficients_file_path: Path):
        mass_per_charge_lookup = MassPerChargeLookup.read_from_file(mass_per_charge_lookup_path)
        mass_coefficients_file_path = MassCoefficientLookup.read_from_csv(mass_coefficients_file_path)
        codice_l2_lo_data = CodiceLoL2SWSpeciesData.read_from_cdf(codice_l2_lo_cdf)
        sw_priority_rates = CodiceLoL1aSWPriorityRates.read_from_cdf(sw_priority_rates_cdf)
        nsw_priority_rates = CodiceLoL1aNSWPriorityRates.read_from_cdf(nsw_priority_rates_cdf)
        codice_l2_direct_events = CodiceLoL2DirectEventData.read_from_cdf(direct_event_path)

        return cls(sw_priority_rates, nsw_priority_rates, codice_l2_lo_data,
                   codice_l2_direct_events, mass_per_charge_lookup,
                   mass_coefficients_file_path)
