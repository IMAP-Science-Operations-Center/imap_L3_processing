from pathlib import Path

import numpy as np
import scipy
import uncertainties
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.special import erf
from uncertainties import correlated_values, ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.constants import BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER, ALPHA_PARTICLE_CHARGE_COULOMBS, ALPHA_PARTICLE_MASS_KG
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_combined_sweeps, \
    get_alpha_peak_indices


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


def alpha_count_rate_model(ev_per_q, density_per_cm3, temperature, bulk_flow_speed_km_per_s):
    density_per_m3 = density_per_cm3 * CENTIMETERS_PER_METER ** 3
    bulk_flow_speed_meters_per_s = bulk_flow_speed_km_per_s * METERS_PER_KILOMETER
    energy = ev_per_q * ALPHA_PARTICLE_CHARGE_COULOMBS
    k = BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    a_eff_cm2 = 3.3e-2 / 1000
    a_eff_m2 = a_eff_cm2 / CENTIMETERS_PER_METER ** 2

    delta_e_over_e = 0.085
    delta_v_over_v = 1 / 2 * delta_e_over_e
    delta_phi_degrees = 30

    m = ALPHA_PARTICLE_MASS_KG
    v_th = np.sqrt(2 * k * temperature / m)
    beta = 1 / v_th ** 2
    v_e = np.sqrt(2 * energy / m)
    result = (density_per_m3 * a_eff_m2 * (beta / np.pi) ** (3 / 2) * np.exp(
        -beta * (v_e ** 2 + bulk_flow_speed_meters_per_s ** 2 - 2 * v_e * bulk_flow_speed_meters_per_s))
              * np.sqrt(np.pi / (beta * bulk_flow_speed_meters_per_s * v_e))
              * erf(np.sqrt(beta * bulk_flow_speed_meters_per_s * v_e) * np.radians(delta_phi_degrees / 2))
              * v_e ** 4 * delta_v_over_v * np.arcsin(v_th / v_e))

    return result


def calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps(
        table: AlphaTemperatureDensityCalibrationTable,
        alpha_sw_speed: ufloat,
        count_rates: uarray,
        energies: ndarray,
):
    average_count_rates, energies = calculate_combined_sweeps(count_rates, energies)

    alpha_particle_peak_slice = get_alpha_peak_indices(average_count_rates, energies)

    peak_energies = energies[alpha_particle_peak_slice]
    peak_average_alpha_count_rates = average_count_rates[alpha_particle_peak_slice]

    initial_parameter_guess = [0.15, 3.6e5, nominal_values(alpha_sw_speed)]
    values, covariance = scipy.optimize.curve_fit(alpha_count_rate_model,
                                                  peak_energies,
                                                  nominal_values(peak_average_alpha_count_rates),
                                                  sigma=std_devs(peak_average_alpha_count_rates),
                                                  absolute_sigma=True,
                                                  bounds=[[0, 0, 0], [np.inf, np.inf, np.inf]],
                                                  p0=initial_parameter_guess)
    residual = abs(alpha_count_rate_model(peak_energies, *values) - nominal_values(peak_average_alpha_count_rates))
    reduced_chisq = np.sum(np.square(residual / std_devs(peak_average_alpha_count_rates))) / (len(peak_energies) - 3)
    if reduced_chisq > 10:
        raise ValueError("Failed to fit - chi-squared too large", reduced_chisq)
    density, temperature, speed = correlated_values(values, covariance)
    density = table.lookup_density(speed, density, temperature)
    temperature = table.lookup_temperature(speed, density, temperature)

    return temperature, density
