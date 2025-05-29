from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _half_spin_to_esa_step_lookup():
    return np.array([
        0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97, 103, 109,
        115, 121, 127
    ])


@dataclass
class GeometricFactorLookup:
    _full_factor: float
    _reduced_factor: float
    _esa_step_end_index: np.ndarray = field(default_factory=_half_spin_to_esa_step_lookup)

    @classmethod
    def read_from_csv(cls, filepath: Path):
        values = np.loadtxt(filepath, dtype=np.float64, skiprows=1, delimiter=',')
        return cls(values[0], values[1])

    def get_geometric_factors(self, rgfo_half_spin: np.ma.masked_array) -> np.ndarray:
        geometric_factors = np.full((rgfo_half_spin.shape[0], 128), self._reduced_factor)
        for epoch_i in range(rgfo_half_spin.shape[0]):
            if rgfo_half_spin[epoch_i] is not np.ma.masked:
                half_spin = int(rgfo_half_spin.data[epoch_i]) - 1
                if half_spin < 0:
                    continue
                last_esa_step = self._esa_step_end_index[half_spin]
                geometric_factors[epoch_i, :last_esa_step + 1] = self._full_factor
            else:
                geometric_factors[epoch_i, :] = np.nan
        return geometric_factors
