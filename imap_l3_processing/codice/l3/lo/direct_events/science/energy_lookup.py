from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyLookup:
    bin_centers: np.ndarray
    bin_edges: np.ndarray

    @property
    def num_bins(self):
        return len(self.bin_centers)

    def get_energy_index(self, energy: np.ndarray | float) -> np.ndarray:
        return np.digitize(energy, self.bin_edges)

    @classmethod
    def from_bin_centers(cls, bin_centers: np.ndarray):
        log_centers = np.log(bin_centers)
        log_bin_deltas = np.diff(log_centers)
        bin_edges = np.exp(log_centers[:-1] + log_bin_deltas / 2)

        return cls(bin_centers=bin_centers, bin_edges=bin_edges)
