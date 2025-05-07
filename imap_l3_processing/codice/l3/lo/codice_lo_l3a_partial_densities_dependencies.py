from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup

SW_SPECIES_DESCRIPTOR = 'sw-species'
MASS_PER_CHARGE_DESCRIPTOR = 'mass-per-charge'


@dataclass
class CodiceLoL3aPartialDensitiesDependencies:
    codice_l2_lo_data: CodiceLoL2SWSpeciesData
    mass_per_charge_lookup: MassPerChargeLookup

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        sectored_intensities_file_paths = []

        for science_input in dependencies.get_science_inputs():
            match science_input:
                case ScienceInput(data_type="l2", source="codice",
                                  descriptor=descriptor) if descriptor == SW_SPECIES_DESCRIPTOR:
                    sectored_intensities_file_paths.extend(science_input.filename_list)

        assert len(sectored_intensities_file_paths) == 1
        sectored_intensities_file_path = sectored_intensities_file_paths[0]

        mass_per_charge_ancillary_file_paths = dependencies.get_file_paths(source='codice',
                                                                           descriptor=MASS_PER_CHARGE_DESCRIPTOR)
        assert len(mass_per_charge_ancillary_file_paths) == 1
        mass_per_charge_ancillary_file_path = mass_per_charge_ancillary_file_paths[0].name

        sectored_intensities_downloaded_path = imap_data_access.download(sectored_intensities_file_path)
        mass_per_charge_ancillary_file_downloaded_path = imap_data_access.download(mass_per_charge_ancillary_file_path)

        return cls.from_file_paths(sectored_intensities_downloaded_path, mass_per_charge_ancillary_file_downloaded_path)

    @classmethod
    def from_file_paths(cls, codice_l2_lo_cdf: Path,
                        mass_per_charge_lookup_path: Path):
        mass_per_charge_lookup = MassPerChargeLookup.read_from_file(mass_per_charge_lookup_path)
        codice_l2_lo_data = CodiceLoL2SWSpeciesData.read_from_cdf(codice_l2_lo_cdf)

        return cls(codice_l2_lo_data,
                   mass_per_charge_lookup)
