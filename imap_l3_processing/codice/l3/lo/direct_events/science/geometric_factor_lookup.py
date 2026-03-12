from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import pandas as pd

import numpy as np


def _half_spin_to_esa_step_lookup():
    return np.array([
        0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97, 103, 109,
        115, 121, 127
    ])

POSITION = TypeVar('POSITION')
ESA_STEP = TypeVar('ESA_STEP')

@dataclass
class GeometricFactorLookup:
    _full_factor: np.ndarray[(POSITION, ESA_STEP)]
    _reduced_factor: np.ndarray[(POSITION, ESA_STEP)]
    _esa_step_end_index: np.ndarray = field(default_factory=_half_spin_to_esa_step_lookup)

    @classmethod
    def read_from_csv(cls, filepath: Path):
        df = (pd.read_csv(filepath)
              .sort_values(['mode', 'esa_step'])
              .groupby('mode'))

        full = df.get_group('full').drop(['mode', 'esa_step'], axis=1).to_numpy()
        reduced = df.get_group('reduced').drop(['mode', 'esa_step'], axis=1).to_numpy()

        return cls(full.T, reduced.T)

    def get_geometric_factors(self, rgfo_half_spin: np.ma.masked_array) -> np.ndarray:
        num_epochs = rgfo_half_spin.shape[0]
        (num_positions, num_esa_steps) = self._reduced_factor.shape

        geometric_factors = np.full((num_epochs, num_positions, num_esa_steps), np.nan)
        for epoch_i in range(rgfo_half_spin.shape[0]):
            if rgfo_half_spin[epoch_i] is not np.ma.masked:
                half_spin_index = int(rgfo_half_spin.data[epoch_i]) - 1
                first_reduced_esa_step = self._esa_step_end_index[half_spin_index] + 1 if half_spin_index >= 0 else 0

                geometric_factors[epoch_i, :, 0:first_reduced_esa_step] = self._full_factor[:, 0:first_reduced_esa_step]
                geometric_factors[epoch_i, :, first_reduced_esa_step:] = self._reduced_factor[:, first_reduced_esa_step:]

        return geometric_factors
