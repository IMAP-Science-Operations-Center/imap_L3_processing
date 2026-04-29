from dataclasses import dataclass
from pathlib import Path

import numpy as np

ESA_TO_ENERGY_PER_CHARGE_LOOKUP_DESCRIPTOR = "lo-energy-per-charge"


@dataclass
class EnergyLookup:
    bin_centers: np.ndarray
    delta_plus: np.ndarray
    delta_minus: np.ndarray

    @classmethod
    def read_from_csv(cls, path_to_csv: Path):
        energy_lowers, energy_centers, energy_uppers = np.loadtxt(path_to_csv, delimiter=',', skiprows=1, usecols=(4, 5, 6)).T

        return cls(bin_centers=energy_centers,
                   delta_minus=energy_centers - energy_lowers,
                   delta_plus=energy_uppers - energy_centers)

    @property
    def num_bins(self):
        return len(self.bin_centers)
