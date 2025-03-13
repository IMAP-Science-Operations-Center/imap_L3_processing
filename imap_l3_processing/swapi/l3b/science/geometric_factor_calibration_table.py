from __future__ import annotations

import numpy as np
from numpy import ndarray


class GeometricFactorCalibrationTable:
    def __init__(self, data: ndarray):
        self.grid = data[:, 0]
        self.geometric_factor_grid = data[:, 1]

    def lookup_geometric_factor(self, energy):
        return np.interp(energy, self.grid, self.geometric_factor_grid)

    @classmethod
    def from_file(cls, file_path) -> GeometricFactorCalibrationTable:
        data = np.loadtxt(file_path, skiprows=1, delimiter=',')
        return cls(data)
