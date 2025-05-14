from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

SPECIES = TypeVar("SPECIES")
AZIMUTH = TypeVar("AZIMUTH")
ENERGIES = TypeVar("ENERGIES")


@dataclass
class EfficiencyLookup:
    efficiency_data: np.ndarray[(SPECIES, AZIMUTH, ENERGIES)]

    @classmethod
    def create_with_fake_data(cls, num_species, num_azimuths, num_energies):
        rng = np.random.default_rng()
        return cls(efficiency_data=rng.random((num_species, num_azimuths, num_energies)))
