from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.hi.l3.models import HiIntensityMapData
from imap_l3_processing.hi.l3.utils import read_hi_l2_data


@dataclass
class HiL3SpectralFitDependencies:
    hi_l3_data: HiIntensityMapData

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HiL3SpectralFitDependencies:
        input_map_filenames = dependencies.get_file_paths(source="hi")

        if len(input_map_filenames) != 1:
            raise ValueError("Missing Hi dependency.")

        hi_l3_file = imap_data_access.download(input_map_filenames[0].name)
        return cls.from_file_paths(hi_l3_file)

    @classmethod
    def from_file_paths(cls, hi_l3_path: Path) -> HiL3SpectralFitDependencies:
        return cls(read_hi_l2_data(hi_l3_path))
