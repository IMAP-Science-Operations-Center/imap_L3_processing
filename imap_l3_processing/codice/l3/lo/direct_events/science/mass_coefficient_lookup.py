from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MassCoefficientLookup:
    coefficients: np.ndarray

    def __getitem__(self, item):
        return self.coefficients[item]

    @classmethod
    def read_from_csv(cls, path: Path):
        mass_coefficients = np.loadtxt(path, dtype=float, delimiter=',')
        return cls(mass_coefficients)
