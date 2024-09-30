from dataclasses import dataclass

import numpy as np

from imap_processing.constants import HYDROGEN_BULK_VELOCITY_IN_KM_PER_SECOND, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS
from imap_processing.spice_wrapper import fake_spice_context
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable


class DensityOfNeutralHeliumLookupTable:
    pass


@dataclass
class PickupIonModelFit:
    geometric_factor_lookup_table: GeometricFactorCalibrationTable
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    energy_center: float
    energy: float
    theta_center: float
    delta_theta: float
    psi_center: float
    delta_center: float

    def f(self, r, w, psi, cooling_index, ionization_rate, cutoff_speed):
        pass
        # first_term = cooling_index / 4.0 * np.pi
        # second_term = ionization_rate * r_e ** 2 / (r * sw_bulk_velocity * cutoff_speed ** 3)
        # w_term = w ** (cooling_index - 3)
        # neutral_helium_density = self.density_of_neutral_helium.lookup(r * w ** cooling_index, psi)
        # step_function = 0 if (1 - w) < 0 else 1
        #
        # return first_term * second_term * w_term * neutral_helium_density * step_function

    def fit_function(self):
        pass


def calculate_pui_energy_cutoff(epoch, bulk_flow_speed):
    with fake_spice_context() as spice:
        imap_velocity_x, imap_velocity_y, imap_velocity_z = spice.spkezr("IMAP", epoch, "HCI", "NONE", "SUN")[3:6]
        imap_velocity = np.sqrt(imap_velocity_x ** 2 + imap_velocity_y ** 2 + imap_velocity_z ** 2)

    proton_velocity_cutoff = bulk_flow_speed - HYDROGEN_BULK_VELOCITY_IN_KM_PER_SECOND - imap_velocity
    return 0.5 * (PROTON_MASS_KG / PROTON_CHARGE_COULOMBS) * (2 * proton_velocity_cutoff) ** 2


def extract_pui_energy_bins(energies, observed_count_rates, energy_cutoff, background_count_rate):
    extracted_energies = []
    count_rates = []

    for energy, count_rate in zip(energies, observed_count_rates):
        if energy > energy_cutoff and count_rate > background_count_rate:
            extracted_energies.append(energy)
            count_rates.append(count_rate)

    return np.array(extracted_energies), np.array(count_rates)
