from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import uncertainties
from numpy import ndarray
from scipy.interpolate import LinearNDInterpolator
from scipy.special import erf
from uncertainties import correlated_values, ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER, ALPHA_PARTICLE_CHARGE_COULOMBS, ALPHA_PARTICLE_MASS_KG, SWAPI_EFFECTIVE_AREA_CM2
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_combined_sweeps, \
    get_alpha_peak_indices
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


class AlphaTemperatureDensityCalibrationTable:
    def __init__(self, lookup_table_array: np.ndarray):
        coords = lookup_table_array[:, [0, 1, 3]]
        values = lookup_table_array[:, [2, 4]]

        self.interp = LinearNDInterpolator(coords, values)

    @classmethod
    def from_file(cls, file_path: Path):
        lookup_table = np.loadtxt(file_path)
        return cls(lookup_table)

    @uncertainties.wrap
    def lookup_temperature(self, sw_speed, fit_density, fit_temperature):
        return self.interp(sw_speed, fit_density, fit_temperature)[1]

    @uncertainties.wrap
    def lookup_density(self, sw_speed, fit_density, fit_temperature):
        return self.interp(sw_speed, fit_density, fit_temperature)[0]


def alpha_count_rate_model(efficiency, ev_per_q, density_per_cm3, temperature, bulk_flow_speed_km_per_s):
    density_per_m3 = density_per_cm3 * CENTIMETERS_PER_METER ** 3
    bulk_flow_speed_meters_per_s = bulk_flow_speed_km_per_s * METERS_PER_KILOMETER
    energy = ev_per_q * ALPHA_PARTICLE_CHARGE_COULOMBS
    k = BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    a_eff_cm2 = efficiency * SWAPI_EFFECTIVE_AREA_CM2
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
        efficiency: float,
):
    average_count_rates, energies = calculate_combined_sweeps(count_rates, energies)

    alpha_particle_peak_slice = get_alpha_peak_indices(average_count_rates, energies)

    peak_energies = energies[alpha_particle_peak_slice]
    peak_average_alpha_count_rates = average_count_rates[alpha_particle_peak_slice]
    at_least_minimum = nominal_values(peak_average_alpha_count_rates) > 0
    filtered_peak_count_rates = peak_average_alpha_count_rates[at_least_minimum]
    filtered_peak_energies = peak_energies[at_least_minimum]

    initial_parameter_guess = [0.15, 3.6e5]

    def model(ev_per_q, density, temperature):
        return alpha_count_rate_model(efficiency, ev_per_q, density, temperature, nominal_values(alpha_sw_speed))

    values, covariance = scipy.optimize.curve_fit(model,
                                                  filtered_peak_energies,
                                                  nominal_values(filtered_peak_count_rates),
                                                  sigma=std_devs(filtered_peak_count_rates),
                                                  absolute_sigma=True,
                                                  bounds=[[0, 0], [np.inf, np.inf]],
                                                  p0=initial_parameter_guess)
    residual = abs(model(filtered_peak_energies, *values) - nominal_values(filtered_peak_count_rates))
    reduced_chisq = np.sum(np.square(residual / std_devs(filtered_peak_count_rates))) / (
            len(filtered_peak_energies) - 2)
    bad_fit_flag = SwapiL3Flags.NONE
    if reduced_chisq > 10:
        bad_fit_flag = SwapiL3Flags.HI_CHI_SQ
    density, temperature = correlated_values(values, covariance)
    density = table.lookup_density(alpha_sw_speed, density, temperature)
    temperature = table.lookup_temperature(alpha_sw_speed, density, temperature)

    return AlphaSolarWindTemperatureAndDensity(temperature, density, bad_fit_flag)


@dataclass
class AlphaSolarWindTemperatureAndDensity:
    temperature: ufloat
    density: ufloat
    bad_fit_flag: int = SwapiL3Flags.NONE
