from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy import ndarray
from spiceypy import spiceypy

from imap_processing.constants import HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, \
    PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000, \
    HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, ONE_AU_IN_KM
from imap_processing.spice_wrapper import fake_spice_context
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection


class DensityOfNeutralHeliumLookupTable:
    @staticmethod
    def density(position: float, psi: float):
        return 1


class InterstellarHeliumInflowLookupTable:
    @staticmethod
    def inflow_direction():
        return 2


@dataclass
class FittingParameters:
    cooling_index: float
    ionization_rate: float
    cutoff_speed: float
    background_count_rate: float


@dataclass
class ForwardModel:
    fitting_params: FittingParameters
    epoch: datetime
    solar_wind_vector_gse_frame: ndarray
    solar_wind_speed_inertial_frame: ndarray

    # at some point we might want to pass in vectors for energy, theta, phi
    def f(self, energy, theta, phi):
        ephemeris_time = spiceypy.datetime2et(self.epoch)
        speed = calculate_sw_speed(PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, energy)  # energy vs E/q?
        pui_vector_instrument_frame = calculate_pui_velocity_vector(speed, theta, phi)
        pui_vector_gse_frame = convert_velocity_relative_to_imap(pui_vector_instrument_frame, "IMAP_SWAPI", "GSE")
        pui_vector_solar_wind_frame = pui_vector_gse_frame - self.solar_wind_vector_gse_frame
        magnitude = np.linalg.norm(pui_vector_solar_wind_frame)
        w = magnitude / self.fitting_params.cutoff_speed
        imap_position_eclip2000_frame_state = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[0:3]
        distance, longitude, latitude = spiceypy.reclat(imap_position_eclip2000_frame_state)
        psi = longitude - 0  # HELIUM_INFLOW_DIRECTION from the lookup table

        neutral_helium_density = DensityOfNeutralHeliumLookupTable().density(
            distance * w ** self.fitting_params.cooling_index,
            psi)
        term1 = self.fitting_params.cooling_index / (4 * np.pi)
        term2 = (self.fitting_params.ionization_rate * ONE_AU_IN_KM ** 2) / (
                distance * self.solar_wind_speed_inertial_frame * self.fitting_params.cutoff_speed)
        term3 = w ** (self.fitting_params.cooling_index - 3)
        term4 = neutral_helium_density
        term5 = np.heaviside(1 - w, 0.5)
        return term1 * term2 * term3 * term4 * term5  # What units should the result be in? What units are the inputs in?


def model_count_rate(fitting_params: FittingParameters, energy_bin_index: int, energy_bin_center: float,
                     response_lookup_table_collection: InstrumentResponseLookupTableCollection,
                     geometric_table: GeometricFactorCalibrationTable) -> float:
    epoch = datetime(2010, 1, 1)

    with fake_spice_context() as spice:
        ephemeris_time = spice.datetime2et(epoch)

        # proton l3a calculation - should l3a be an input for l3b? or redo the part of the calculation we need?
        # solar wind speed, flow deflection angle, clock angle
        # at 1-minute intervals - need to average 10 minutes in some way
        # convert each minute into xyz vector?
        v, deflection, clock = l3a_calculation()
        solar_wind_vector_instrument_frame = convert_flow_deflection_to_xyz(v, deflection, clock)
        solar_wind_vector_gse_frame = convert_velocity(solar_wind_vector_instrument_frame, "IMAP_RTN",
                                                       "GSE")  # account for IMAP velocity?
        solar_wind_vector_inertial_frame = convert_velocity(solar_wind_vector_instrument_frame, "IMAP_RTN",
                                                            "HCI")  # account for IMAP velocity?
        solar_wind_speed_inertial_frame = np.linalg.norm(solar_wind_vector_inertial_frame)

        response_lookup_table = response_lookup_table_collection.get_table_for_energy_bin(energy_bin_index)
        integral = 0
        for i in range(len(response_lookup_table.response)):
            speed = calculate_sw_speed(PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS,
                                       response_lookup_table.energy[i])
            colatitude = 90 - response_lookup_table.elevation[i]
            integral += response_lookup_table.response[i] * f(response_lookup_table.energy[i], colatitude,
                                                              response_lookup_table.azimuth[i]) \
                        * speed ** 4 * response_lookup_table.d_energy[i] * np.cos(np.deg2rad(colatitude)) \
                        * response_lookup_table.d_azimuth[i] * response_lookup_table.d_elevation[i]

        geometric_factor = geometric_table.lookup_geometric_factor(energy_bin_center)

        return (geometric_factor / 2) * integral / _model_count_rate_denominator(
            response_lookup_table) + fitting_params.background_count_rate


def _model_count_rate_denominator(response_lookup_table: InstrumentResponseLookupTable) -> float:
    colatitude_radians = np.deg2rad(90 - response_lookup_table.elevation)

    rows = response_lookup_table.d_energy * np.cos(
        colatitude_radians) * response_lookup_table.d_elevation * response_lookup_table.d_azimuth

    return rows.sum()


def convert_velocity_relative_to_imap(velocity, ephemeris_time, from_frame, to_frame):
    imap_rotation_matrix = spiceypy.sxform(from_frame, to_frame, ephemeris_time)
    state = [0, 0, 0, *velocity]
    state_in_target_frame = np.matmul(imap_rotation_matrix, state)
    velocity_in_target_frame_relative_to_imap = state_in_target_frame[3:6]
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, to_frame, "NONE", "SUN")[3:6]

    return velocity_in_target_frame_relative_to_imap + imap_velocity


def calculate_pui_velocity_vector(sw_speed: float, colatitude: float, azimuth: float) -> np.ndarray:
    return spiceypy.sphrec(sw_speed, colatitude, azimuth)


def calculate_pui_energy_cutoff(epoch, sw_velocity_in_imap_frame):
    ephemeris_time = spiceypy.datetime2et(epoch)
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[
                    3:6]
    solar_wind_velocity = convert_velocity_relative_to_imap(
        sw_velocity_in_imap_frame, ephemeris_time, "IMAP", "ECLIPJ2000")
    hydrogen_velocity = spiceypy.latrec(-HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND,
                                        HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000,
                                        HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000)

    proton_velocity_cutoff_vector = solar_wind_velocity - hydrogen_velocity - imap_velocity
    proton_speed_cutoff = np.linalg.norm(proton_velocity_cutoff_vector)
    return 0.5 * (PROTON_MASS_KG / PROTON_CHARGE_COULOMBS) * (2 * proton_speed_cutoff) ** 2


def extract_pui_energy_bins(energy_bin_labels, energies, observed_count_rates, energy_cutoff, background_count_rate):
    extracted_energy_bins = []
    count_rates = []
    extracted_energy_bin_labels = []

    for label, energy, count_rate in zip(energy_bin_labels, energies, observed_count_rates):
        if energy > energy_cutoff and count_rate > background_count_rate:
            extracted_energy_bins.append(energy)
            count_rates.append(count_rate)
            extracted_energy_bin_labels.append(label)

    return np.array(extracted_energy_bin_labels), np.array(extracted_energy_bins), np.array(count_rates)
