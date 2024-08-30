import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.special import erf
from uncertainties import correlated_values
from uncertainties.unumpy import uarray, nominal_values

from imap_processing import constants
from imap_processing.constants import PROTON_MASS_KG, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import get_proton_peak_indices, \
    calculate_sw_speed_h_plus
from imap_processing.swapi.l3a.science.speed_calculation import find_peak_center_of_mass_index, interpolate_energy, \
    extract_coarse_sweep


def proton_count_rate_model(ev_per_q, density_per_cm3, temperature, bulk_flow_speed_km_per_s):
    density_per_m3 = density_per_cm3 * CENTIMETERS_PER_METER**3
    bulk_flow_speed_meters_per_s = bulk_flow_speed_km_per_s * METERS_PER_KILOMETER
    energy = ev_per_q * constants.PROTON_CHARGE_COULOMBS
    k = BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    a_eff_cm2 = 3.3e-2 / 1000
    # TODO verify these units
    a_eff_m2 = a_eff_cm2 / CENTIMETERS_PER_METER**2

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


def calculate_proton_solar_wind_temperature_and_density(coincident_count_rates: uarray, energy):
    coincident_count_rates = extract_coarse_sweep(coincident_count_rates)
    energy = extract_coarse_sweep(energy)
    proton_peak_indices = get_proton_peak_indices(coincident_count_rates)

    initial_speed_guess = calculate_proton_speed_from_one_sweep(coincident_count_rates, energy, proton_peak_indices)

    initial_parameter_guess = [5, 1e5, nominal_values(initial_speed_guess)]
    values, covariance = scipy.optimize.curve_fit(proton_count_rate_model,
                                                  energy[proton_peak_indices],
                                                  nominal_values(coincident_count_rates[proton_peak_indices]),
                                                  p0=initial_parameter_guess)
    density, temperature, speed = correlated_values(values, covariance)

    return temperature, density


def calculate_proton_speed_from_one_sweep(coincident_count_rates, energy, proton_peak_indices):
    center_of_mass_index = find_peak_center_of_mass_index(proton_peak_indices, coincident_count_rates)
    energy_at_com = interpolate_energy(center_of_mass_index, energy)
    initial_speed_guess = calculate_sw_speed_h_plus(energy_at_com)
    return initial_speed_guess


def demo(density=5.0, temp=1e5, speed=450):

    voltage = np.geomspace(100, 19000, 62)

    plt.loglog(voltage, proton_count_rate_model(voltage, density, temp, speed))
    plt.ylim(1e-3, 1e8)

    plt.show()
