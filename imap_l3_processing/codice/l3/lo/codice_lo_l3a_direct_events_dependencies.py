from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup, \
    ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2DirectEventData, CodiceLoL1aSWPriorityRates, \
    CodiceLoL1aNSWPriorityRates

SW_PRIORITY_DESCRIPTOR = 'lo-sw-priority'
NSW_PRIORITY_DESCRIPTOR = 'lo-nsw-priority'
DIRECT_EVENTS_DESCRIPTOR = 'lo-direct-events'
MASS_COEFFICIENT_DESCRIPTOR = 'mass-coefficient-lookup'


@dataclass
class CodiceLoL3aDirectEventsDependencies:
    codice_lo_l1a_sw_priority_rates: CodiceLoL1aSWPriorityRates
    codice_lo_l1a_nsw_priority_rates: CodiceLoL1aNSWPriorityRates
    codice_l2_direct_events: CodiceLoL2DirectEventData
    mass_coefficient_lookup: MassCoefficientLookup
    energy_lookup: EnergyLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        sw_priority_rates_paths = []
        nsw_priority_rates_paths = []
        direct_events_paths = []
        for science_input in dependencies.get_science_inputs():
            match science_input:
                case ScienceInput(data_type="l1a", source="codice",
                                  descriptor=descriptor) if descriptor == SW_PRIORITY_DESCRIPTOR:
                    sw_priority_rates_paths.extend(science_input.filename_list)
                case ScienceInput(data_type="l1a", source="codice",
                                  descriptor=descriptor) if descriptor == NSW_PRIORITY_DESCRIPTOR:
                    nsw_priority_rates_paths.extend(science_input.filename_list)
                case ScienceInput(data_type="l2", source="codice",
                                  descriptor=descriptor) if descriptor == DIRECT_EVENTS_DESCRIPTOR:
                    direct_events_paths.extend(science_input.filename_list)

        assert len(sw_priority_rates_paths) == 1
        sw_priority_rates_path = sw_priority_rates_paths[0]
        assert len(nsw_priority_rates_paths) == 1
        nsw_priority_rates_path = nsw_priority_rates_paths[0]
        assert len(direct_events_paths) == 1
        direct_events_path = direct_events_paths[0]

        [mass_coefficients_file_path] = dependencies.get_file_paths(source='codice',
                                                                    descriptor=MASS_COEFFICIENT_DESCRIPTOR)
        [energy_per_charge_file_path] = dependencies.get_file_paths(source='codice',
                                                                    descriptor=ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR)

        sw_priority_rates_downloaded_path = download(sw_priority_rates_path)
        nsw_priority_rates_downloaded_path = download(nsw_priority_rates_path)
        direct_events_downloaded_path = download(direct_events_path)

        mass_coefficients_file_downloaded_path = download(mass_coefficients_file_path.name)
        energy_per_charge_lookup_donwloaded_path = download(energy_per_charge_file_path.name)

        return cls.from_file_paths(
            sw_priority_rates_downloaded_path,
            nsw_priority_rates_downloaded_path,
            direct_events_downloaded_path,
            mass_coefficients_file_downloaded_path,
            energy_per_charge_lookup_donwloaded_path
        )

    @classmethod
    def from_file_paths(cls, sw_priority_rates_cdf: Path, nsw_priority_rates_cdf: Path,
                        direct_event_path: Path,
                        mass_coefficients_file_path: Path,
                        esa_to_energy_per_charge_file_path: Path):
        mass_coefficients_file_path = MassCoefficientLookup.read_from_csv(mass_coefficients_file_path)
        sw_priority_rates = CodiceLoL1aSWPriorityRates.read_from_cdf(sw_priority_rates_cdf)
        nsw_priority_rates = CodiceLoL1aNSWPriorityRates.read_from_cdf(nsw_priority_rates_cdf)
        codice_l2_direct_events = CodiceLoL2DirectEventData.read_from_cdf(direct_event_path)
        energy_lookup = EnergyLookup.read_from_csv(esa_to_energy_per_charge_file_path)

        return cls(sw_priority_rates, nsw_priority_rates, codice_l2_direct_events, mass_coefficients_file_path,
                   energy_lookup)
