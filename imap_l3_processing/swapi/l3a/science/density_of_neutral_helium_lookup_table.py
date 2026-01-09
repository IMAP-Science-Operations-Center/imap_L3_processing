from typing import Union

import numpy as np
from numpy import ndarray
from scipy.interpolate import RegularGridInterpolator


class DensityOfNeutralHeliumLookupTable:
    def __init__(self, calibration_table: np.ndarray):
        angle = np.unique(calibration_table[:, 0])
        distance = np.unique(calibration_table[:, 1])

        self.grid = (angle, distance)
        values_shape = tuple(len(x) for x in self.grid)
        self.densities = np.ascontiguousarray(
            calibration_table[:, 2].reshape(values_shape)
        )
        if 360.0 not in angle:
            assert angle[0] == 0.0, "expected table to start at angle 0"
            angle_with_360 = np.append(angle, 360.0)
            self.grid = (angle_with_360, distance)
            self.densities = np.append(self.densities, self.densities[0:1], axis=0)

        self._interp = RegularGridInterpolator(self.grid, self.densities, bounds_error=False, fill_value=0,
                                               method='linear')

    def density(self, angle: Union[ndarray, float], distance: Union[ndarray, float]):
        if isinstance(distance, float):
            coords = np.array((angle % 360, distance))
            return self._interp(coords)[0]
        else:
            coords = np.empty((len(distance), 2))
            coords[:, 0] = angle % 360
            coords[:, 1] = distance
            return self._interp(coords)

    @classmethod
    def from_file(cls, file):
        data = np.loadtxt(file)
        return cls(data)

    def get_minimum_distance(self) -> float:
        return self.grid[1][0]
