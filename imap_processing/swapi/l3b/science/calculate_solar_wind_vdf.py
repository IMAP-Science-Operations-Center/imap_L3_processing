from __future__ import annotations

import numpy as np
from numpy import ndarray

from imap_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, ALPHA_PARTICLE_CHARGE_COULOMBS, \
    ALPHA_PARTICLE_MASS_KG, PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed


def calculate_vdf(particle_mass, particle_charge, energies: ndarray, average_count_rates: ndarray,
                  efficiency: float, geometric_factor_table: GeometricFactorCalibrationTable):
    velocities = calculate_sw_speed(particle_mass, particle_charge, energies)
    geometric_factors = geometric_factor_table.lookup_geometric_factor(energies)

    proton_mass_per_charge = particle_mass / particle_charge
    numerator = 4 * np.pi * proton_mass_per_charge * average_count_rates
    denominator = (energies * geometric_factors * efficiency)
    probabilities = numerator / denominator

    return velocities, probabilities


def calculate_proton_solar_wind_vdf(energies: ndarray, average_count_rates: ndarray,
                                    efficiency: float, geometric_factor_table: GeometricFactorCalibrationTable):
    return calculate_vdf(PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, energies, average_count_rates, efficiency,
                         geometric_factor_table)


def calculate_alpha_solar_wind_vdf(energies: ndarray, average_count_rates: ndarray,
                                   efficiency: float, geometric_factor_table: GeometricFactorCalibrationTable):
    return calculate_vdf(ALPHA_PARTICLE_MASS_KG, ALPHA_PARTICLE_CHARGE_COULOMBS, energies,
                         average_count_rates, efficiency,
                         geometric_factor_table)


def calculate_pui_solar_wind_vdf(energies: ndarray, average_count_rates: ndarray, efficiency: float,
                                 geometric_factor_table: GeometricFactorCalibrationTable):
    return calculate_vdf(PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, energies,
                         average_count_rates, efficiency,
                         geometric_factor_table)


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
