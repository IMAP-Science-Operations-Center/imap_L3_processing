from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EnergyLookup:
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    delta_plus: np.ndarray
    delta_minus: np.ndarray

    @classmethod
    def read_from_csv(cls, path_to_csv: Path):
        indices, energy_lowers, energy_centers, energy_uppers = np.loadtxt(path_to_csv, delimiter=',', skiprows=1).T

        return cls(bin_centers=energy_centers,
                   bin_edges=energy_uppers,
                   delta_minus=energy_centers - energy_lowers,
                   delta_plus=energy_uppers - energy_centers)

    @property
    def num_bins(self):
        return len(self.bin_centers)

    def get_energy_index(self, energy: np.ndarray | float) -> np.ndarray:
        return np.digitize(energy, self.bin_edges)

    @classmethod
    def from_bin_centers(cls, bin_centers: np.ndarray):
        log_centers = np.log(bin_centers)
        log_bin_deltas = np.diff(log_centers)

        average_log_delta = np.array([np.average(log_bin_deltas)])

        lower_edges = np.exp(log_centers - np.concatenate([average_log_delta, log_bin_deltas]) / 2.0)
        upper_edges = np.exp(log_centers + np.concatenate([log_bin_deltas, average_log_delta]) / 2.0)

        bin_edges = np.exp(log_centers[:-1] + log_bin_deltas / 2)

        return cls(bin_centers=bin_centers,
                   bin_edges=bin_edges,
                   delta_plus=upper_edges - bin_centers,
                   delta_minus=bin_centers - lower_edges)
