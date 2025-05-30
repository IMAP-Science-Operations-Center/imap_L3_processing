from typing import Union

import numpy as np
import scipy
from numpy import ndarray


class DensityOfNeutralHeliumLookupTable:
    def __init__(self, calibration_table: np.ndarray):
        angle = np.unique(calibration_table[:, 0])
        distance = np.unique(calibration_table[:, 1])

        self.grid = (angle, distance)
        values_shape = tuple(len(x) for x in self.grid)

        self.densities = calibration_table[:, 2].reshape(values_shape)

    def density(self, angle: Union[ndarray, float], distance: Union[ndarray, float]):
        if isinstance(distance, float):
            coords = np.array((angle % 360, distance))
            result = scipy.interpolate.interpn(self.grid, self.densities,
                                               coords, bounds_error=False, fill_value=0)
            return result[0]
        else:
            coords = np.empty((len(distance), 2))
            coords[:, 0] = angle % 360
            coords[:, 1] = distance
            return scipy.interpolate.interpn(self.grid, self.densities,
                                             coords, bounds_error=False, fill_value=0)

    @classmethod
    def from_file(cls, file):
        data = np.loadtxt(file)
        return cls(data)

    def get_minimum_distance(self) -> float:
        return self.grid[1][0]
