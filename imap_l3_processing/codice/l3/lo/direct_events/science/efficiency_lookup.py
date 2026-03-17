from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
import pandas as pd

import numpy as np

POSITION = TypeVar("POSITION")
ENERGIES = TypeVar("ENERGIES")

@dataclass
class EfficiencyLookup:
    efficiency_data: np.ndarray[(POSITION, ENERGIES)]

    @classmethod
    def read_from_csv(cls, path: Path, species: str) -> EfficiencyLookup:
        df = pd.read_csv(path)
        efficiency_data = (df[(df['species'] == species) & (df['product'] == 'sw')]
                           .sort_values('esa_step')
                           .drop(['species', 'product', 'esa_step'], axis=1)
                           .to_numpy())

        return cls(efficiency_data=efficiency_data.T)
