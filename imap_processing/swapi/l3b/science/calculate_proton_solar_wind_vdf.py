from __future__ import annotations

import numpy as np
from numpy import ndarray

from imap_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed_h_plus


def calculate_proton_solar_wind_vdf(energies, count_rates, efficiency, geometric_factor):
    velocities = calculate_sw_speed_h_plus(energies)

    proton_mass_per_charge = PROTON_MASS_KG / PROTON_CHARGE_COULOMBS
    numerator = 4 * np.pi * proton_mass_per_charge * count_rates
    denominator = (energies * geometric_factor * efficiency)
    probabilities = numerator / denominator

    return velocities, probabilities


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
