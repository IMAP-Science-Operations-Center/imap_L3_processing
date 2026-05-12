from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_SPIN_SECTORS, CODICE_LO_NUM_AZIMUTH_BINS, \
    CODICE_LO_NUM_ESA_STEPS


def _half_spin_to_esa_step_lookup():
    return np.array([
        0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97, 103, 109,
        115, 121, 127
    ])


POSITION = TypeVar('POSITION')
ESA_STEP = TypeVar('ESA_STEP')


@dataclass
class GeometricFactorLookup:
    _full_factor: np.ndarray[(ESA_STEP, POSITION)]
    _reduced_factor: np.ndarray[(ESA_STEP, POSITION)]
    _esa_step_end_index: np.ndarray = field(default_factory=_half_spin_to_esa_step_lookup)

    @classmethod
    def read_from_csv(cls, filepath: Path):
        df = (pd.read_csv(filepath)
              .sort_values(['mode', 'esa_step'])
              .groupby('mode'))

        full = df.get_group('full').drop(['mode', 'esa_step'], axis=1).to_numpy()
        reduced = df.get_group('reduced').drop(['mode', 'esa_step'], axis=1).to_numpy()

        return cls(full, reduced)

    def get_geometric_factors(self,
                              rgfo_half_spin: np.ma.masked_array,
                              rgfo_spin_sector: np.ma.masked_array,
                              rgfo_esa_step: np.ma.masked_array,
                              half_spin: np.ma.masked_array,
                              ) -> np.ndarray:
        num_epochs = half_spin.shape[0]

        full_shape = (num_epochs, CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_SPIN_SECTORS,
                      CODICE_LO_NUM_AZIMUTH_BINS)
        full_geometric_factors_full_shape = np.broadcast_to(self._full_factor[np.newaxis, :, np.newaxis, :],
                                                            full_shape
                                                            ).copy()
        reduced_geometric_factors_full_shape = np.broadcast_to(self._reduced_factor[np.newaxis, :, np.newaxis, :],
                                                               full_shape
                                                               ).copy()

        half_spin_greater_than_rgfo_half_spin = half_spin > rgfo_half_spin[:, np.newaxis]
        half_spin_equal_to_rgfo_half_spin = half_spin == rgfo_half_spin[:, np.newaxis]
        spin_sector_greater_than_rgfo_spin_sector = np.arange(0, CODICE_LO_NUM_SPIN_SECTORS)[np.newaxis,
                                                    :] > rgfo_spin_sector[:, np.newaxis]
        equal_half_spin_and_greater_spin_sector = (half_spin_equal_to_rgfo_half_spin[:, :, np.newaxis] &
                                                   spin_sector_greater_than_rgfo_spin_sector[:, np.newaxis, :])
        spin_sector_equal_to_rgfo_spin_sector = np.arange(0, CODICE_LO_NUM_SPIN_SECTORS)[np.newaxis,
                                                :] == rgfo_spin_sector[:, np.newaxis]

        esa_step_greater_than_rgfo_esa_step = np.arange(0, CODICE_LO_NUM_ESA_STEPS)[np.newaxis, :] > rgfo_esa_step[:,
                                                                                                     np.newaxis]
        equal_half_spin_equal_spin_sector_and_greater_esa_step = (
                    half_spin_equal_to_rgfo_half_spin[:, :, np.newaxis] & spin_sector_equal_to_rgfo_spin_sector[:,
                                                                          np.newaxis,
                                                                          :] & esa_step_greater_than_rgfo_esa_step[:, :,
                                                                               np.newaxis])
        needing_reduced_factor = half_spin_greater_than_rgfo_half_spin[:, :, np.newaxis,
                                 np.newaxis] | equal_half_spin_and_greater_spin_sector[:, :, :,
                                               np.newaxis] | equal_half_spin_equal_spin_sector_and_greater_esa_step[:,
                                                             :, :, np.newaxis]

        return np.where(
            needing_reduced_factor,
            reduced_geometric_factors_full_shape,
            full_geometric_factors_full_shape)
