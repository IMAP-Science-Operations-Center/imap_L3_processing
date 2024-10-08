from dataclasses import dataclass
from datetime import datetime

import numpy as np
import scipy
from spiceypy import reclat

from imap_processing.constants import HYDROGEN_BULK_VELOCITY_IN_KM_PER_SECOND, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, \
    PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS
from imap_processing.spice_wrapper import fake_spice_context
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed, \
    calculate_sw_speed_h_plus
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection


class DensityOfNeutralHeliumLookupTable:
    pass


@dataclass
class FittingParameters:
    cooling_index: float
    ionization_rate: float
    cutoff_speed: float
    background_count_rate: float


@dataclass
class Integrand:
    cooling_index: float
    ionization_rate: float
    cutoff_speed: float

    def f(self, r, w, psi):
        sw_bulk_velocity = 450
        r_e = 1
        first_term = self.cooling_index / 4.0 * np.pi
        second_term = self.ionization_rate * r_e ** 2 / (r * sw_bulk_velocity * self.cutoff_speed ** 3)
        w_term = w ** (self.cooling_index - 3)
        # neutral_helium_density = self.density_of_neutral_helium.lookup(r * w ** cooling_index, psi)
        neutral_helium_density = 0.1
        step_function = 0 if (1 - w) < 0 else 1

        return first_term * second_term * w_term * neutral_helium_density * step_function

    def integrand(self, energy, theta, phi):
        velocity = calculate_sw_speed(PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, energy)
        return self.f(velocity, theta, phi)


@dataclass
class PickupIonModelFit:
    geometric_factor_lookup_table: GeometricFactorCalibrationTable
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    theta_center: float
    delta_theta: float
    phi_center: float
    delta_phi: float
    energies_and_bounds: dict

    @classmethod
    def create_fitting(cls, geometric_factor_lookup, density_of_neutral_helium, energies):
        lower_energies = energies + (scipy.ndimage.shift(energies, -1) - energies) / 2.0
        lower_energies[-1] = energies[-1]

        upper_energies = energies + (scipy.ndimage.shift(energies, 1) - energies) / 2.0
        upper_energies[0] = energies[0]

        energies_and_bounds = {}
        for energy, lower_bound, upper_bound in zip(energies, lower_energies, upper_energies):
            energies_and_bounds[energy] = (lower_bound, upper_bound)

        return cls(geometric_factor_lookup, DensityOfNeutralHeliumLookupTable(), 0, 10, 0, 30, energies_and_bounds)

    def fit_function(self, energies, cooling_index: float, ionization_rate: float, cutoff_speed: float):
        modeled_count_rates = []

        integrand_closure = Integrand(cooling_index, ionization_rate, cutoff_speed)

        for energy in energies:
            modeled_count_rates.append(model_count_rate)

        print(modeled_count_rates)


def model_count_rate(fitting_params: FittingParameters, energy_bin_index: int, energy_bin_center: float,
                     response_lookup_table_collection: InstrumentResponseLookupTableCollection, geometric_table: GeometricFactorCalibrationTable) -> float:
    epoch = datetime(2010, 1, 1)
    with fake_spice_context() as spice:
        imap_x, imap_y, imap_z, = spice.spkezr("IMAP", epoch, "GSE", "NONE", "SUN")[0:3]
        imap_velocity = spice.spkezr("IMAP", epoch, "GSE", "NONE", "SUN")[3:6]

        ephemeris_time = 0.0054

        def convert_velocity(velocity, from_frame, to_frame)
            imap_rotation_matrix = spice.sxform(from_frame, to_frame, ephemeris_time)
            state = [0, 0, 0, *velocity]
            state_in_target_frame = np.matmul(imap_rotation_matrix, state)
            return state_in_target_frame[3:6]

        # proton l3a calculation - should l3a be an input for l3b? or redo the part of the calculation we need?
        # solar wind speed, flow deflection angle, clock angle
        # at 1-minute intervals - need to average 10 minutes in some way
        # convert each minute into xyz vector?
        v, deflection, clock = l3a_calculation()
        solar_wind_vector_instrument_frame = convert_flow_deflection_to_xyz(v, deflection, clock)
        solar_wind_vector_gse_frame = convert_velocity(solar_wind_vector_instrument_frame, "IMAP_RTN", "GSE") # account for IMAP velocity?
        solar_wind_vector_inertial_frame = convert_velocity(solar_wind_vector_instrument_frame, "IMAP_RTN", "HCI") # account for IMAP velocity?
        solar_wind_speed_inertial_frame = np.linalg.norm(solar_wind_vector_inertial_frame)


        # at some point we might want to pass in vectors for energy, theta, phi
        def f(energy, theta, phi):
            speed = calculate_sw_speed_h_plus(energy)  # energy vs E/q?
            pui_vector_instrument_frame = convert_from_instrument_frame_angles(speed, theta, phi)
            pui_vector_gse_frame = convert_velocity(pui_vector_instrument_frame, "IMAP_SWAPI", "GSE")
            pui_vector_solar_wind_frame = (pui_vector_gse_frame + imap_velocity) - solar_wind_vector_gse_frame
            magnitude = np.linalg.norm(pui_vector_solar_wind_frame)
            w = magnitude / fitting_params.cutoff_speed
            imap_position_eclip2000_frame_state = spice.spkezr("IMAP", epoch, "ECLIPJ2000", "NONE", "SUN")[0:3]
            distance, longitude, latitude = reclat(imap_position_eclip2000_frame_state)
            psi = longitude - HELIUM_INFLOW_DIRECTION
            lut = DensityOfNeutralHeliumLookupTable()

            neutral_helium_density = lut.density(distance * w ** fitting_params.cooling_index,  psi)
            term1 =  fitting_params.cooling_index / (4 * np.pi)
            term2 = (fitting_params.ionization_rate * ONE_AU_IN_KM ** 2) / (distance * solar_wind_speed_inertial_frame * fitting_params.cutoff_speed)
            term3 = w ** (fitting_params.cooling_index -3)
            term4 = neutral_helium_density
            term5 = np.heaviside(1 - w, 0.5)
            return term1 * term2 * term3 * term4 * term5  # What units should the result be in? What units are the inputs in?

        response_lookup_table = response_lookup_table_collection.get_table_for_energy_bin(energy_bin_index)
        integral = 0
        for i in range(len(response_lookup_table.response)):
            speed = calculate_sw_speed_h_plus(response_lookup_table.energy[i])
            colatitude = 90-response_lookup_table.elevation[i]
            integral += response_lookup_table.response * f(response_lookup_table.energy[i],colatitude, response_lookup_table.azimuth[i]) \
                * speed**4 * response_lookup_table.d_energy * np.cos(np.deg2rad(colatitude)) \
                * response_lookup_table.d_azimuth[i] * response_lookup_table.d_elevation[i]


        geometric_factor = geometric_table.lookup_geometric_factor(energy_bin_center)

        return (geometric_factor/2) * integral / _model_count_rate_denominator(response_lookup_table) + fitting_params.background_count_rate



def _model_count_rate_denominator(response_lookup_table: InstrumentResponseLookupTable) -> float:
    colatitude_radians = np.deg2rad(90 - response_lookup_table.elevation)

    rows = response_lookup_table.d_energy * np.cos(
        colatitude_radians) * response_lookup_table.d_elevation * response_lookup_table.d_azimuth

    return rows.sum()


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
