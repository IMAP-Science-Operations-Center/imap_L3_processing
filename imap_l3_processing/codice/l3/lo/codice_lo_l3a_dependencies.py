from dataclasses import dataclass
from pathlib import Path

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.utils import download_dependency_from_path


@dataclass
class CodiceLoL3aDependencies:
    codice_l2_lo_data: CodiceLoL2Data
    mass_per_charge_lookup: MassPerChargeLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        for dep in dependencies.get_science_inputs():
            if dep.data_type != 'l2':
                dependencies.processing_input.remove(dep)

        science_file_paths = dependencies.get_file_paths(source='codice', descriptor='sectored-intensities')
        ancillary_file_paths = dependencies.get_file_paths(source='codice',
                                                           descriptor='mass-per-charge-lookup')

        for file_path in [*science_file_paths, *ancillary_file_paths]:
            download_dependency_from_path(file_path)

        return cls.from_file_paths(science_file_paths[0], ancillary_file_paths[0])

    @classmethod
    def from_file_paths(cls, codice_l2_lo_cdf: Path, mass_per_charge_lookup_path: Path):
        mass_per_charge_lookup_path = MassPerChargeLookup.read_from_file(mass_per_charge_lookup_path)
        codice_l2_lo_cdf = CodiceLoL2Data.read_from_cdf(codice_l2_lo_cdf)
