import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from scipy.special import erf
from uncertainties.unumpy import uarray

from imap_processing import constants
from imap_processing.constants import PROTON_MASS_KG, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN


def count_rate_fitting(ev_per_q, density, temperature, bulk_flow_speed):
    energy = ev_per_q*constants.PROTON_CHARGE_COULOMBS
    k = BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
    a_eff = 3.3e-2 / 1000

    delta_e_over_e = 0.085
    delta_v_over_v = 1/2 * delta_e_over_e
    delta_phi_degrees = 30

    m = PROTON_MASS_KG
    v_th = np.sqrt(2*k*temperature / m)
    beta = 1 / v_th**2
    v_e = np.sqrt(2*energy / m)

    return (density * a_eff * (beta / np.pi)**(3/2) * np.exp(-beta*(v_e**2+bulk_flow_speed**2-2*v_e*bulk_flow_speed))
    * np.sqrt(np.pi / (beta * bulk_flow_speed * v_e))
     * erf(np.sqrt(beta * bulk_flow_speed * v_e)*np.radians(delta_phi_degrees/2))
     * v_e**4* delta_v_over_v * np.arcsin(v_th/v_e))

density = 5e6
temp = 1e5
speed = 450e3


voltage = np.geomspace(100, 19000, 62)

plt.loglog(voltage, count_rate_fitting(voltage, density, temp, speed))
plt.ylim(1e-3,1e8)

plt.show()

def calculate_proton_solar_wind_temperature_and_density(coincident_count_rates: uarray, energy):
    scipy.optimize.curve_fit()