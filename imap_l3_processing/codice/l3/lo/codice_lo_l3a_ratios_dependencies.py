from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.models import CodiceLoPartialDensityData

PARTIAL_DENSITY_DESCRIPTOR = 'lo-partial-densities'


@dataclass
class CodiceLoL3aRatiosDependencies:
    partial_density_data: CodiceLoPartialDensityData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        input_file_paths = dependencies.get_file_paths(source='codice', descriptor=PARTIAL_DENSITY_DESCRIPTOR)
        assert len(input_file_paths) == 1
        input_file_name = input_file_paths[0].name

        downloaded_file = imap_data_access.download(input_file_name)

        return cls.from_file_paths(downloaded_file)

    @classmethod
    def from_file_paths(cls, codice_l2_lo_cdf: Path | str):
        partial_density_data = CodiceLoPartialDensityData.read_from_cdf(codice_l2_lo_cdf)

        return cls(partial_density_data)
