from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

SPECIES = TypeVar("SPECIES")
AZIMUTH = TypeVar("AZIMUTH")
ENERGIES = TypeVar("ENERGIES")


@dataclass
class EfficiencyLookup:
    efficiency_data: np.ndarray[(AZIMUTH, ENERGIES)]

    @classmethod
    def read_from_csv(cls, path: Path) -> EfficiencyLookup:
        return cls(efficiency_data=np.loadtxt(path, delimiter=",", skiprows=1).T)
