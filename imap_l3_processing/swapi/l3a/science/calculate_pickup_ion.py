from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy import ndarray
from scipy.optimize import OptimizeResult
from uncertainties.unumpy import uarray

from imap_l3_processing.constants import HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND, PROTON_MASS_KG, PROTON_CHARGE_COULOMBS, \
    HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS, HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000, \
    HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, ONE_AU_IN_KM, HELIUM_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, \
    METERS_PER_KILOMETER, CENTIMETERS_PER_METER, ONE_SECOND_IN_NANOSECONDS, BOLTZMANN_CONSTANT_JOULES_PER_KELVIN
from imap_l3_processing.spice_wrapper import spiceypy
from imap_l3_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_combined_sweeps
from imap_l3_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed
from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from imap_l3_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_l3_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection


def calculate_pickup_ion_values(instrument_response_lookup_table, geometric_factor_calibration_table,
                                energy: np.ndarray[float],
                                count_rates: uarray, center_of_epoch: int,
                                background_count_rate_cutoff: float, sw_velocity_vector: ndarray,
                                density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable) -> FittingParameters:
    ephemeris_time = spiceypy.unitim(center_of_epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    sw_velocity = np.linalg.norm(sw_velocity_vector)

    initial_guess = np.array([1.5, 1e-7, sw_velocity, 0.1])
    energy_labels = range(62, 0, -1)
    energy_cutoff = calculate_pui_energy_cutoff(ephemeris_time, sw_velocity_vector)
    average_count_rates, energies = calculate_combined_sweeps(count_rates, energy)

    extracted_energy_labels, extracted_energies, extracted_count_rates = extract_pui_energy_bins(energy_labels,
                                                                                                 energies,
                                                                                                 average_count_rates,
                                                                                                 energy_cutoff,
                                                                                                 background_count_rate_cutoff)
    model_count_rate_calculator = ModelCountRateCalculator(instrument_response_lookup_table,
                                                           geometric_factor_calibration_table, sw_velocity_vector,
                                                           density_of_neutral_helium_lookup_table)
    indices = list(zip(extracted_energy_labels, extracted_energies))

    result: OptimizeResult = scipy.optimize.minimize(
        calc_chi_squared, initial_guess,
        bounds=((1.0, 5.0), (0.6e-7, 2.1e-7),
                (sw_velocity * .8, sw_velocity * 1.2), (0, 0.2)),
        args=(extracted_count_rates, indices, model_count_rate_calculator,
              ephemeris_time),
        method='Nelder-Mead',
        options=dict(disp=True))
    return FittingParameters(*result.x)


def calculate_helium_pui_density(epoch: int,
                                 sw_velocity_vector: ndarray,
                                 density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                                 fitting_params: FittingParameters) -> float:
    ephemeris_time = spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    model = build_forward_model(fitting_params, ephemeris_time, sw_velocity_vector,
                                density_of_neutral_helium_lookup_table)
    lower_discontinuity = (density_of_neutral_helium_lookup_table.get_minimum_distance() / (
            model.distance_km / ONE_AU_IN_KM)) ** (
                                  1 / fitting_params.cooling_index) * fitting_params.cutoff_speed
    points = (0, lower_discontinuity, fitting_params.cutoff_speed)

    results = scipy.integrate.quad(lambda v: model.f(v) * v * v, 0, fitting_params.cutoff_speed, limit=100,
                                   points=points)
    return 4 * np.pi * results[0] / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3


def calculate_helium_pui_temperature(epoch: int,
                                     sw_velocity_vector: ndarray,
                                     density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable,
                                     fitting_params: FittingParameters) -> float:
    ephemeris_time = spiceypy.unitim(epoch / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
    model = build_forward_model(fitting_params, ephemeris_time, sw_velocity_vector,
                                density_of_neutral_helium_lookup_table)
    lower_discontinuity = (density_of_neutral_helium_lookup_table.get_minimum_distance() / (
            model.distance_km / ONE_AU_IN_KM)) ** (
                                  1 / fitting_params.cooling_index) * fitting_params.cutoff_speed
    points = (0, lower_discontinuity, fitting_params.cutoff_speed)

    numerator = scipy.integrate.quad(lambda v: model.f(v) * v ** 4, 0, fitting_params.cutoff_speed, points=points,
                                     limit=100)
    denominator = scipy.integrate.quad(lambda v: model.f(v) * v ** 2, 0, fitting_params.cutoff_speed, points=points,
                                       limit=100)
    return HE_PUI_PARTICLE_MASS_KG / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN) * \
        numerator[0] / denominator[0] * \
        METERS_PER_KILOMETER ** 2


@dataclass
class FittingParameters:
    cooling_index: float
    ionization_rate: float
    cutoff_speed: float
    background_count_rate: float


@dataclass
class ForwardModel:
    fitting_params: FittingParameters
    ephemeris_time: float
    solar_wind_vector_eclipj2000_frame: ndarray
    solar_wind_speed_inertial_frame: float
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable
    distance_km: float
    psi: float

    def compute_from_instrument_frame(self, speed, theta, phi):
        pui_vector_instrument_frame = calculate_pui_velocity_vector(speed, theta, phi)
        pui_vector_eclipj2000_frame = convert_velocity_relative_to_imap(pui_vector_instrument_frame,
                                                                        self.ephemeris_time,
                                                                        "IMAP_SWAPI", "ECLIPJ2000")
        pui_vector_solar_wind_frame = pui_vector_eclipj2000_frame - self.solar_wind_vector_eclipj2000_frame
        pickup_ion_speed = np.linalg.norm(pui_vector_solar_wind_frame, axis=-1)

        result = self.f(pickup_ion_speed)
        return result

    def f(self, pickup_ion_speed):
        w = pickup_ion_speed / self.fitting_params.cutoff_speed
        radius_in_au = self.distance_km / ONE_AU_IN_KM
        neutral_helium_density_per_cm3 = self.density_of_neutral_helium_lookup_table.density(
            self.psi, radius_in_au * w ** self.fitting_params.cooling_index)
        neutral_helium_density_per_km3 = neutral_helium_density_per_cm3 * (
                CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        term1 = self.fitting_params.cooling_index / (4 * np.pi)
        term2 = (self.fitting_params.ionization_rate * ONE_AU_IN_KM ** 2) / (
                self.distance_km * self.solar_wind_speed_inertial_frame * self.fitting_params.cutoff_speed ** 3)
        term3 = w ** (self.fitting_params.cooling_index - 3)
        term4 = neutral_helium_density_per_km3
        term5 = np.heaviside(1 - w, 0.5)
        return term1 * term2 * term3 * term4 * term5


def build_forward_model(fitting_params: FittingParameters, ephemeris_time: float, solar_wind_vector: ndarray,
                        density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable) -> ForwardModel:
    solar_wind_vector_eclipj2000_frame = convert_velocity_relative_to_imap(solar_wind_vector,
                                                                           ephemeris_time,
                                                                           "IMAP_DPS",
                                                                           "ECLIPJ2000")
    imap_position_eclip2000_frame_state = spiceypy.spkezr(
        "IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[0][0:3]
    distance_km, longitude, latitude = spiceypy.reclat(imap_position_eclip2000_frame_state)
    psi = np.rad2deg(longitude) - HELIUM_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000

    return ForwardModel(fitting_params, ephemeris_time, solar_wind_vector_eclipj2000_frame,
                        np.linalg.norm(solar_wind_vector_eclipj2000_frame),
                        density_of_neutral_helium_lookup_table, distance_km, psi)


@dataclass
class ModelCountRateCalculator:
    response_lookup_table_collection: InstrumentResponseLookupTableCollection
    geometric_table: GeometricFactorCalibrationTable
    solar_wind_vector: np.ndarray
    density_of_neutral_helium_lookup_table: DensityOfNeutralHeliumLookupTable

    def model_count_rate(self, indices_and_energy_centers: list[tuple[int, float]],
                         fitting_params: FittingParameters, ephemeris_time: float) -> np.ndarray:
        forward_model = build_forward_model(fitting_params, ephemeris_time, self.solar_wind_vector,
                                            self.density_of_neutral_helium_lookup_table)
        model_count_rates = []
        for energy_bin_index, energy_bin_center in indices_and_energy_centers:
            model_count_rates.append(
                self.model_one_count_rate(energy_bin_index, energy_bin_center, forward_model)
            )
        return np.array(model_count_rates)

    def model_one_count_rate(self, energy_bin_index, energy_bin_center, forward_model) -> float:
        response_lookup_table = self.response_lookup_table_collection.get_table_for_energy_bin(energy_bin_index)
        integral = model_count_rate_integral(response_lookup_table, forward_model)

        geometric_factor = self.geometric_table.lookup_geometric_factor(energy_bin_center)
        denominator = _model_count_rate_denominator(response_lookup_table)
        return (geometric_factor / 2) * integral / denominator + forward_model.fitting_params.background_count_rate


def calc_chi_squared(fit_params_array: np.ndarray, observed_count_rates: np.ndarray,
                     indices_and_energy_centers: list[tuple[int, float]], calculator: ModelCountRateCalculator,
                     ephemeris_time: float):
    cooling_index, ionization_rate, cutoff_speed, background_count_rate = fit_params_array
    fit_params = FittingParameters(cooling_index, ionization_rate, cutoff_speed, background_count_rate)
    modeled_rates = calculator.model_count_rate(indices_and_energy_centers, fit_params, ephemeris_time)

    result = 2 * sum(
        modeled_rates - observed_count_rates + observed_count_rates * np.log(observed_count_rates / modeled_rates))
    return result


def model_count_rate_integral(response_lookup_table: InstrumentResponseLookupTable, forward_model: ForwardModel):
    speed = calculate_sw_speed(HE_PUI_PARTICLE_MASS_KG, PUI_PARTICLE_CHARGE_COULOMBS,
                               response_lookup_table.energy)
    count_rates = forward_model.compute_from_instrument_frame(speed, response_lookup_table.elevation,
                                                              response_lookup_table.azimuth)

    integrals = response_lookup_table.response * count_rates \
                * speed ** 4 * \
                response_lookup_table.d_energy * np.cos(np.deg2rad(response_lookup_table.elevation)) * \
                response_lookup_table.d_azimuth * response_lookup_table.d_elevation

    return np.sum(integrals)


def _model_count_rate_denominator(response_lookup_table: InstrumentResponseLookupTable) -> float:
    elevation_radians = np.deg2rad(response_lookup_table.elevation)

    rows = response_lookup_table.d_energy * np.cos(
        elevation_radians) * response_lookup_table.d_elevation * response_lookup_table.d_azimuth

    return rows.sum()


def convert_velocity_to_reference_frame(velocity: ndarray, ephemeris_time: float, from_frame: str,
                                        to_frame: str) -> ndarray:
    rotation_matrix = spiceypy.sxform(from_frame, to_frame, ephemeris_time)

    state = velocity[..., np.newaxis]

    state_in_target_frame = np.matmul(rotation_matrix[3:6, 3:6], state)
    return state_in_target_frame[..., 0]


def convert_velocity_relative_to_imap(velocity, ephemeris_time, from_frame, to_frame):
    velocity_in_target_frame_relative_to_imap = convert_velocity_to_reference_frame(velocity, ephemeris_time,
                                                                                    from_frame, to_frame)
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, to_frame, "NONE", "SUN")[0][3:6]

    return velocity_in_target_frame_relative_to_imap + imap_velocity


def calculate_velocity_vector(sw_speed: ndarray, elevation: ndarray, azimuth: ndarray) -> np.ndarray:
    elevation_radians = np.deg2rad(elevation)
    azimuth_radians = np.deg2rad(azimuth)
    z = sw_speed * np.sin(elevation_radians)
    xy_radius = sw_speed * np.cos(elevation_radians)
    x = xy_radius * np.cos(azimuth_radians)
    y = xy_radius * np.sin(azimuth_radians)
    return np.transpose([x, y, z])


def calculate_pui_velocity_vector(speed: ndarray, elevation: ndarray, azimuth: ndarray) -> np.ndarray:
    y_axis_azimuth = 90
    return calculate_velocity_vector(-speed, elevation, y_axis_azimuth - azimuth)


def calculate_pui_energy_cutoff(ephemeris_time: float, sw_velocity_in_imap_frame):
    imap_velocity = spiceypy.spkezr("IMAP", ephemeris_time, "ECLIPJ2000", "NONE", "SUN")[0][
                    3:6]
    solar_wind_velocity = convert_velocity_relative_to_imap(
        sw_velocity_in_imap_frame, ephemeris_time, "IMAP_DPS", "ECLIPJ2000")
    hydrogen_velocity = spiceypy.latrec(-HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND,
                                        HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000,
                                        HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000)

    proton_velocity_cutoff_vector = solar_wind_velocity - hydrogen_velocity - imap_velocity
    proton_speed_cutoff = np.linalg.norm(proton_velocity_cutoff_vector)
    return 0.5 * (PROTON_MASS_KG / PROTON_CHARGE_COULOMBS) * (2 * proton_speed_cutoff * METERS_PER_KILOMETER) ** 2


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


def calculate_solar_wind_velocity_vector(speeds: ndarray, deflection_angle: ndarray, clock_angle: ndarray) -> ndarray:
    elevation_angle = 90 - deflection_angle
    clock_angle_origin_in_despun_frame = -90
    return calculate_velocity_vector(-speeds, elevation_angle, clock_angle + clock_angle_origin_in_despun_frame)


def calculate_ten_minute_velocities(speeds: ndarray, deflection_angle: ndarray, clock_angle: ndarray) -> ndarray:
    velocity_vector = calculate_solar_wind_velocity_vector(speeds, deflection_angle, clock_angle)
    left_slice = 0
    chunked_velocities = []
    while left_slice < len(velocity_vector):
        chunked_velocities.append(np.mean(velocity_vector[left_slice:left_slice + 10], axis=0))
        left_slice += 10
    return np.array(chunked_velocities)
