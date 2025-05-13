import numpy as np


class SpinAngleLookup:
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    bin_deltas: np.ndarray

    def __init__(self):
        self.bin_edges = np.linspace(0, 360, 24, endpoint=False)
        self.bin_deltas = np.diff(self.bin_edges)
        assert np.all(self.bin_deltas == self.bin_deltas[0])

        self.bin_centers = self.bin_edges + (self.bin_deltas[0] / 2)

    def get_spin_angle_index(self, spin_angle: np.ndarray | float):
        return np.digitize(spin_angle, self.bin_edges) - 1

    @property
    def num_bins(self):
        return len(self.bin_centers)


class ElevationLookup:

    def __init__(self):
        self.bin_centers = np.linspace(0, 180, 13)
        self.bin_deltas = np.diff(self.bin_centers)
        assert np.all(self.bin_deltas == self.bin_deltas[0])

        self.bin_edges = self.bin_centers - (self.bin_deltas[0] / 2)

    def get_elevation_index(self, elevation: np.ndarray | float):
        return np.digitize(elevation, self.bin_edges) - 1

    @property
    def num_bins(self):
        return len(self.bin_centers)
