import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import EventDirection


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


class PositionToElevationLookup:

    def __init__(self):
        self.bin_centers = np.linspace(0, 180, 13)
        self.bin_deltas = np.diff(self.bin_centers)
        assert np.all(self.bin_deltas == self.bin_deltas[0])

        self.bin_edges = self.bin_centers - (self.bin_deltas[0] / 2)

        self.elevation_indices_by_apd = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    @property
    def num_bins(self):
        return len(self.bin_centers)

    def get_elevation_index(self, elevation: np.ndarray | float):
        return np.digitize(elevation, self.bin_edges) - 1

    def event_direction_for_apd(self, apd: int) -> EventDirection:
        return EventDirection.Sunward if self.apd_to_elevation(apd) < 37.5 else EventDirection.NonSunward

    def apd_to_elevation(self, apd: int) -> float:
        return float(self.bin_centers[self.apd_to_elevation_index(apd)])

    def apd_to_elevation_index(self, apd: int) -> int:
        return self.elevation_indices_by_apd[apd - 1]
