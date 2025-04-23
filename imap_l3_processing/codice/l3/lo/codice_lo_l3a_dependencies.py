from dataclasses import dataclass
from pathlib import Path

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data, CodiceLoL2bPriorityRates, CodiceLoL2DirectEventData
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.esa_step_lookup import ESAStepLookup
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.utils import download_dependency_from_path


@dataclass
class CodiceLoL3aDependencies:
    codice_l2_lo_data: CodiceLoL2Data
    codice_l2b_lo_priority_rates: CodiceLoL2bPriorityRates
    codice_l2_direct_events: CodiceLoL2DirectEventData
    mass_per_charge_lookup: MassPerChargeLookup
    esa_steps_lookup: ESAStepLookup
    mass_coefficient_lookup: MassCoefficientLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        for dep in dependencies.get_science_inputs():
            if dep.data_type[:2] != 'l2':
                dependencies.processing_input.remove(dep)

        priority_rates_paths = dependencies.get_file_paths(source='codice', descriptor='priority-rates')
        sectored_intensities_file_paths = dependencies.get_file_paths(source='codice',
                                                                      descriptor='sectored-intensities')
        direct_events_path = dependencies.get_file_paths(source='codice', descriptor='direct-events')
        mass_per_charge_ancillary_file_path = dependencies.get_file_paths(source='codice',
                                                                          descriptor='mass-per-charge-lookup')
        esa_step_ancillary_file_path = dependencies.get_file_paths(source='codice',
                                                                   descriptor='esa-step-lookup')

        mass_coefficients_file_path = dependencies.get_file_paths(source='codice',
                                                                  descriptor='mass-coefficient-lookup')

        for file_path in [*priority_rates_paths, *sectored_intensities_file_paths, *direct_events_path,
                          *mass_per_charge_ancillary_file_path,
                          *esa_step_ancillary_file_path, *mass_coefficients_file_path]:
            download_dependency_from_path(file_path)

        # TODO: get the actual path from instrument team/algorithm doc

        return cls.from_file_paths(sectored_intensities_file_paths[0],
                                   priority_rates_paths[0],
                                   direct_events_path[0],
                                   mass_per_charge_ancillary_file_path[0],
                                   esa_step_ancillary_file_path[0],
                                   mass_coefficients_file_path[0])

    @classmethod
    def from_file_paths(cls, codice_l2_lo_cdf: Path, priority_rates_cdf: Path, direct_event_path: Path,
                        mass_per_charge_lookup_path: Path,
                        esa_step_lookup_path: Path,
                        mass_coefficients_file_path: Path):
        mass_per_charge_lookup = MassPerChargeLookup.read_from_file(mass_per_charge_lookup_path)
        esa_steps_lookup = ESAStepLookup.read_from_file(esa_step_lookup_path)
        mass_coefficients_file_path = MassCoefficientLookup.read_from_csv(mass_coefficients_file_path)
        codice_l2_lo_data = CodiceLoL2Data.read_from_cdf(codice_l2_lo_cdf)
        codice_l2b_priority_rates = CodiceLoL2bPriorityRates.read_from_cdf(priority_rates_cdf)
        codice_l2_direct_events = CodiceLoL2DirectEventData.read_from_cdf(direct_event_path)

        return cls(codice_l2_lo_data, codice_l2b_priority_rates, codice_l2_direct_events, mass_per_charge_lookup,
                   esa_steps_lookup, mass_coefficients_file_path)
