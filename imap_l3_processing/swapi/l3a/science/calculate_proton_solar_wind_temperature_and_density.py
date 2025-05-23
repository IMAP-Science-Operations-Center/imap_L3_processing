import numpy as np
import scipy
import uncertainties
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.special import erf
from uncertainties import correlated_values
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing import constants
from imap_l3_processing.constants import PROTON_MASS_KG, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import get_proton_peak_indices, \
    calculate_sw_speed_h_plus
from imap_l3_processing.swapi.l3a.science.speed_calculation import find_peak_center_of_mass_index, interpolate_energy, \
    extract_coarse_sweep


def proton_count_rate_model(ev_per_q, density_per_cm3, temperature, bulk_flow_speed_km_per_s):
    density_per_m3 = density_per_cm3 * CENTIMETERS_PER_METER ** 3
    bulk_flow_speed_meters_per_s = bulk_flow_speed_km_per_s * METERS_PER_KILOMETER
    energy = ev_per_q * constants.PROTON_CHARGE_COULOMBS
    k = BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    a_eff_cm2 = 3.3e-2 / 1000
    a_eff_m2 = a_eff_cm2 / CENTIMETERS_PER_METER ** 2

    delta_e_over_e = 0.085
    delta_v_over_v = 1 / 2 * delta_e_over_e
    delta_phi_degrees = 30

    m = PROTON_MASS_KG
    v_th = np.sqrt(2 * k * temperature / m)
    beta = 1 / v_th ** 2
    v_e = np.sqrt(2 * energy / m)
    result = (density_per_m3 * a_eff_m2 * (beta / np.pi) ** (3 / 2) * np.exp(
        -beta * (v_e ** 2 + bulk_flow_speed_meters_per_s ** 2 - 2 * v_e * bulk_flow_speed_meters_per_s))
              * np.sqrt(np.pi / (beta * bulk_flow_speed_meters_per_s * v_e))
              * erf(np.sqrt(beta * bulk_flow_speed_meters_per_s * v_e) * np.radians(delta_phi_degrees / 2))
              * v_e ** 4 * delta_v_over_v * np.arcsin(v_th / v_e))

    return result


def calculate_proton_solar_wind_temperature_and_density_for_one_sweep(coincident_count_rates: uarray, energy: ndarray):
    coincident_count_rates = extract_coarse_sweep(coincident_count_rates)
    energy = extract_coarse_sweep(energy)
    proton_peak_indices = get_proton_peak_indices(coincident_count_rates)

    initial_speed_guess = calculate_proton_speed_from_one_sweep(coincident_count_rates, energy, proton_peak_indices)

    initial_parameter_guess = [5, 1e5, nominal_values(initial_speed_guess)]
    peak_energies = energy[proton_peak_indices]
    peak_count_rates = coincident_count_rates[proton_peak_indices]
    values, covariance = scipy.optimize.curve_fit(proton_count_rate_model,
                                                  peak_energies,
                                                  nominal_values(peak_count_rates),
                                                  sigma=std_devs(peak_count_rates),
                                                  absolute_sigma=True,
                                                  bounds=[[0, 0, 0], [np.inf, np.inf, np.inf]],
                                                  p0=initial_parameter_guess)
    residual = abs(proton_count_rate_model(peak_energies, *values) - nominal_values(peak_count_rates))
    reduced_chisq = np.sum(np.square(residual / std_devs(peak_count_rates))) / (len(peak_energies) - 3)
    if reduced_chisq > 10:
        raise ValueError("Failed to fit - chi-squared too large", reduced_chisq)
    density, temperature, speed = correlated_values(values, covariance)

    return temperature, density


def calculate_uncalibrated_proton_solar_wind_temperature_and_density(coincident_count_rates: uarray, energy: ndarray):
    temperatures_per_sweep = []
    densities_per_sweep = []
    for sweep in coincident_count_rates:
        temperature, density = calculate_proton_solar_wind_temperature_and_density_for_one_sweep(sweep, energy)
        temperatures_per_sweep.append(temperature)
        densities_per_sweep.append(density)

    average_temp = np.average(temperatures_per_sweep, weights=1 / std_devs(temperatures_per_sweep) ** 2)
    average_density = np.average(densities_per_sweep, weights=1 / std_devs(densities_per_sweep) ** 2)

    return average_temp, average_density


def calculate_proton_speed_from_one_sweep(coincident_count_rates, energy, proton_peak_indices):
    center_of_mass_index = find_peak_center_of_mass_index(proton_peak_indices, coincident_count_rates)
    energy_at_com = interpolate_energy(center_of_mass_index, energy)
    initial_speed_guess = calculate_sw_speed_h_plus(energy_at_com)
    return initial_speed_guess


class ProtonTemperatureAndDensityCalibrationTable:
    def __init__(self, lookup_table_array: ndarray):
        solar_wind_speed = np.unique(lookup_table_array[:, 0])
        deflection_angle = np.unique(lookup_table_array[:, 1])
        clock_angle = np.unique(lookup_table_array[:, 2])
        fit_density = np.unique(lookup_table_array[:, 3])
        fit_temperature = np.unique(lookup_table_array[:, 5])
        self.grid = (solar_wind_speed, deflection_angle, clock_angle, fit_density, fit_temperature)
        values_shape = tuple(len(x) for x in self.grid)

        self.density_grid = lookup_table_array[:, 4].reshape(values_shape)
        self.temperature_grid = lookup_table_array[:, 6].reshape(values_shape)

    @uncertainties.wrap
    def calibrate_density(self, solar_wind_speed, deflection_angle, clock_angle, fit_density, fit_temperature):
        return scipy.interpolate.interpn(self.grid, self.density_grid,
                                         [solar_wind_speed, deflection_angle, clock_angle % 360, fit_density,
                                          fit_temperature])[0]

    @uncertainties.wrap
    def calibrate_temperature(self, solar_wind_speed, deflection_angle, clock_angle, fit_density, fit_temperature):
        return scipy.interpolate.interpn(self.grid, self.temperature_grid,
                                         [solar_wind_speed, deflection_angle, clock_angle % 360, fit_density,
                                          fit_temperature])[0]

    @classmethod
    def from_file(cls, file_path):
        lookup_table_array = np.loadtxt(file_path)
        return cls(lookup_table_array)


def calculate_proton_solar_wind_temperature_and_density(lookup_table: ProtonTemperatureAndDensityCalibrationTable,
                                                        proton_solar_wind_speed, deflection_angle,
                                                        clock_angle, coincident_count_rates: uarray, energy: ndarray):
    temperature, density = calculate_uncalibrated_proton_solar_wind_temperature_and_density(coincident_count_rates,
                                                                                            energy)
    calibrated_temperature = lookup_table.calibrate_temperature(proton_solar_wind_speed, deflection_angle, clock_angle,
                                                                density,
                                                                temperature)
    calibrated_density = lookup_table.calibrate_density(proton_solar_wind_speed, deflection_angle, clock_angle, density,
                                                        temperature)
    return calibrated_temperature, calibrated_density


def demo(density=5.0, temp=1e5, speed=450):
    voltage = np.geomspace(100, 19000, 62)

    plt.loglog(voltage, proton_count_rate_model(voltage, density, temp, speed))
    plt.ylim(1e-3, 1e8)

    plt.show()
