import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import EventDirection


class SpinAngleLookup:
    bin_centers: np.ndarray
    lower_bin_edges: np.ndarray
    bin_deltas: np.ndarray

    def __init__(self):
        num_bins = 24

        self.lower_bin_edges = np.linspace(0, 360, num_bins, endpoint=False)
        bin_widths = np.diff(self.lower_bin_edges)
        assert np.all(bin_widths == bin_widths[0])
        bin_delta = bin_widths[0] / 2.0

        self.bin_deltas = np.repeat(bin_delta, num_bins)
        self.bin_centers = self.lower_bin_edges + bin_delta

    def get_spin_angle_index(self, spin_angle: np.ndarray | float):
        return np.digitize(spin_angle, self.lower_bin_edges) - 1

    @property
    def num_bins(self):
        return len(self.bin_centers)


class PositionToElevationLookup:
    bin_centers: np.ndarray
    bin_deltas: np.ndarray
    lower_bin_edges: np.ndarray

    def __init__(self):
        self.bin_centers = np.linspace(0, 180, 13)
        bin_half_width = np.diff(self.bin_centers) / 2
        assert np.all(bin_half_width == bin_half_width[0])
        self.bin_deltas = np.repeat(bin_half_width[0], 13)

        self.lower_bin_edges = self.bin_centers - self.bin_deltas

        self.elevation_indices_by_apd = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    @property
    def num_bins(self):
        return len(self.bin_centers)

    def get_elevation_index(self, elevation: np.ndarray | float):
        return np.digitize(elevation, self.lower_bin_edges) - 1

    def event_direction_for_apd(self, apd: int) -> EventDirection:
        return EventDirection.Sunward if self.apd_to_elevation(apd) < 37.5 else EventDirection.NonSunward

    def apd_to_elevation(self, apd: int) -> float:
        return float(self.bin_centers[self.apd_to_elevation_index(apd)])

    def apd_to_elevation_index(self, apd: np.ndarray | int) -> int:
        return self.elevation_indices_by_apd[apd - 1]
