from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_SPIN_SECTORS, CODICE_LO_NUM_ESA_STEPS


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
        use_reduced = self._is_past_rgfo(rgfo_half_spin, rgfo_spin_sector, rgfo_esa_step, half_spin)
        return np.where(
            use_reduced,
            self._reduced_factor[np.newaxis, :, np.newaxis, :],
            self._full_factor[np.newaxis, :, np.newaxis, :],
        )

    @staticmethod
    def _is_past_rgfo(rgfo_half_spin: np.ma.masked_array,
                      rgfo_spin_sector: np.ma.masked_array,
                      rgfo_esa_step: np.ma.masked_array,
                      half_spin: np.ma.masked_array,
                      ) -> np.ndarray:
        rgfo_half_spin_e = rgfo_half_spin[:, None, None, None]
        rgfo_esa_step_e = rgfo_esa_step[:, None, None, None]
        rgfo_spin_sector_mod = (rgfo_spin_sector % 12)[:, None, None, None]

        half_spin_e = half_spin[:, :, None, None]
        esa_step_axis = np.arange(CODICE_LO_NUM_ESA_STEPS)[None, :, None, None]
        spin_sector_axis_mod = np.arange(CODICE_LO_NUM_SPIN_SECTORS)[None, None, :, None] % 12

        half_spin_past = half_spin_e > rgfo_half_spin_e
        half_spin_match = half_spin_e == rgfo_half_spin_e
        spin_sector_past = spin_sector_axis_mod > rgfo_spin_sector_mod
        spin_sector_match = spin_sector_axis_mod == rgfo_spin_sector_mod
        esa_step_past = esa_step_axis > rgfo_esa_step_e

        return (
                half_spin_past
                | (half_spin_match & spin_sector_past)
                | (half_spin_match & spin_sector_match & esa_step_past)
        )
