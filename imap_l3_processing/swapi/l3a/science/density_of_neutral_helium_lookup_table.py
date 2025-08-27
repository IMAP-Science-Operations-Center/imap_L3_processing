from typing import Union

import numpy as np
import scipy
from numpy import ndarray


class DensityOfNeutralHeliumLookupTable:
    def __init__(self, calibration_table: np.ndarray):
        angle = np.unique(calibration_table[:, 0])
        distance = np.unique(calibration_table[:, 1])

        self.angle = np.ascontiguousarray(angle)
        self.distance = np.ascontiguousarray(distance)
        self.grid = (angle, distance)
        values_shape = tuple(len(x) for x in self.grid)

        # self.densities = calibration_table[:, 2].reshape(values_shape)
        self.densities = np.ascontiguousarray(
            calibration_table[:, 2].reshape(values_shape)
        )
        from scipy.interpolate import RegularGridInterpolator
        self._slope_deltas = np.diff(self.distance)
        self._slope_along_intercepts = np.diff(self.densities, axis=1) / self._slope_deltas[np.newaxis, :]
        self._intercept_along_distance = np.diff(
            self.densities[:, :-1] - self._slope_along_intercepts * self.distance[:-1][np.newaxis, :])

        self._interp = RegularGridInterpolator(self.grid, self.densities, bounds_error=False, fill_value=0, method='linear')

    def angle_bracket_indices_and_weights(self, angle_degree: float):
        angle_degree = float(angle_degree) % 360
        angle_grid = self.angle
        n_angles = angle_grid.size

        upper_index = int(np.searchsorted(angle_grid, angle_degree, side='right'))
        lower_index = (upper_index - 1) % n_angles

        lower_angle_value = angle_grid[lower_index]
        upper_angle_value = angle_grid[upper_index]

        if lower_angle_value == upper_angle_value:
            angle_weight = 0.0
        else:
            angle_weight = (angle_degree - lower_angle_value) / (upper_angle_value - lower_angle_value)

        return lower_index, upper_index, angle_weight

    def interpolate_along_distance_for_row(self, angle_row_index, distance):
        distance_grid = self.distance
        slopes = self._slope_along_intercepts[angle_row_index]
        intercepts = self._intercept_along_distance[angle_row_index]

        segment_index = np.searchsorted(distance_grid, distance, side='right') - 1
        in_range_mask = (segment_index >= 0) & (segment_index < distance_grid.size - 1)

        interpolated = np.zeros_like(distance)
        if np.any(in_range_mask):
            k = segment_index[in_range_mask]
            interpolated[in_range_mask] = slopes[k] * distance[in_range_mask] + intercepts[np.minimum(k,len(intercepts)-1)]

        return interpolated


    def density(self, angle: Union[ndarray, float], distance: Union[ndarray, float]):
        if isinstance(distance, float):
            coords = np.array((angle % 360, distance))
            lower_index, upper_index, angle_weight = self.angle_bracket_indices_and_weights(angle)
            # lower angle = custom_interpolate_along_distance_for_row(lower_index,
            # upper angle = custom_interpolate_along_distance_for_row(lower_index,
            # result = scipy.interpolate.interpn(self.grid, self.densities,
            #                                    coords, bounds_error=False, fill_value=0)
            return self._interp(coords)[0]
        else:
            coords = np.empty((len(distance), 2))
            coords[:, 0] = angle % 360
            coords[:, 1] = distance
            return self._interp(coords)

            lower_index, upper_index, angle_weight = self.angle_bracket_indices_and_weights(angle)
            lower_angle = self.interpolate_along_distance_for_row(lower_index, distance)
            upper_angle = self.interpolate_along_distance_for_row(lower_index, distance)

            return (1.0 - angle_weight) * lower_angle + angle_weight * upper_angle

            return scipy.interpolate.interpn(self.grid, self.densities,
                                             coords, bounds_error=False, fill_value=0)

    @classmethod
    def from_file(cls, file):
        data = np.loadtxt(file)
        return cls(data)

    def get_minimum_distance(self) -> float:
        return self.grid[1][0]
