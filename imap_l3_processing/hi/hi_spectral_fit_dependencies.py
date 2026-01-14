from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imap_data_access
import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.maps.map_models import RectangularIntensityMapData, SpectralIndexDependencies


@dataclass
class HiSpectralIndexDependencies(SpectralIndexDependencies):

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection) -> HiSpectralIndexDependencies:
        input_map_filenames = dependencies.get_file_paths(source="hi", data_type="l2") + \
                              dependencies.get_file_paths(source="hi", data_type="l3")

        if len(input_map_filenames) != 1:
            raise ValueError("Missing Hi dependency.")

        hi_l3_file = imap_data_access.download(input_map_filenames[0].name)
        return cls.from_file_paths(hi_l3_file)

    @classmethod
    def from_file_paths(cls, hi_l3_path: Path) -> HiSpectralIndexDependencies:
        return cls(RectangularIntensityMapData.read_from_path(hi_l3_path))

    def get_fit_energy_ranges(self) -> np.ndarray:
        energy_min = np.min(
            self.map_data.intensity_map_data.energy - self.map_data.intensity_map_data.energy_delta_minus)
        energy_max = np.max(
            self.map_data.intensity_map_data.energy + self.map_data.intensity_map_data.energy_delta_plus)

        return np.array([[energy_min, energy_max]])
