from pathlib import Path

import numpy as np
import scipy
import uncertainties


class AlphaTemperatureDensityCalibrationTable:
    def __init__(self, lookup_table_array: np.ndarray):
        solar_wind_speed = np.unique(lookup_table_array[:, 0])
        fit_density = np.unique(lookup_table_array[:, 1])
        fit_temperature = np.unique(lookup_table_array[:, 3])
        self.grid = (solar_wind_speed, fit_density, fit_temperature)
        values_shape = tuple(len(x) for x in self.grid)

        self.density_grid = lookup_table_array[:, 2].reshape(values_shape)
        self.temperature_grid = lookup_table_array[:, 4].reshape(values_shape)

    @classmethod
    def from_file(cls, file_path: Path):
        lookup_table = np.loadtxt(file_path)
        return cls(lookup_table)

    @uncertainties.wrap
    def lookup_temperature(self, sw_speed, density, temperature):
        return scipy.interpolate.interpn(self.grid, self.temperature_grid, [sw_speed, density, temperature])[0]

    @uncertainties.wrap
    def lookup_density(self, sw_speed, density, temperature):
        return scipy.interpolate.interpn(self.grid, self.density_grid, [sw_speed, density, temperature])[0]