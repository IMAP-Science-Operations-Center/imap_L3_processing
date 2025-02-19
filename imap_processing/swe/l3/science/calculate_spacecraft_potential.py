import numpy as np
from scipy.optimize import curve_fit

from imap_processing.constants import ELECTRON_MASS_KG, PROTON_CHARGE_COULOMBS, METERS_PER_KILOMETER


def piece_wise_model(x: np.ndarray, b0: float, b1: float,
                     b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    return np.log(np.piecewise(x, [x <= b2, (x > b2) & (x <= b4), x > b4],
                               [
                                   lambda x: b0 * np.exp(-b1 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(-b3 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(b4 * (b5 - b3)) * np.exp(-b5 * x),
                               ]))


def find_breakpoints(energies: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_flux = np.log(flux)
    slope = -np.diff(log_flux) / np.diff(energies)
    xsratio = slope[1:] / slope[:-1]
    numb = np.max(np.nonzero(xsratio > 0.55), initial=0)

    energies = energies[:numb]
    log_flux = log_flux[:numb]

    b5 = (log_flux[10] - log_flux[11]) / (energies[11] - energies[10])
    b3 = (log_flux[5] - log_flux[6]) / (energies[6] - energies[5])
    b1 = (log_flux[0] - log_flux[1]) / (energies[1] - energies[0])
    b0 = np.exp(log_flux[0] + b1 * energies[0])
    initial_spacecraft_potential = 10
    initial_core_halo_break_point = 80
    initial_guesses = (b0, b1, initial_spacecraft_potential, b3, initial_core_halo_break_point, b5)

    fit, covariance = curve_fit(piece_wise_model, energies, log_flux, initial_guesses)
    return fit[2], fit[4]


def average_flux(flux_data: np.ndarray, geometric_weights: np.ndarray) -> np.ndarray:
    weighted_average = np.sum(flux_data * geometric_weights, axis=-1) / np.sum(geometric_weights)
    return np.mean(weighted_average, axis=1)


def compute_look_directions(inst_el: np.ndarray, inst_az: np.ndarray) -> np.ndarray:
    inst_az_rad = np.deg2rad(inst_az)[..., np.newaxis]
    inst_el_rad = np.deg2rad(inst_el)
    z = np.sin(inst_el_rad)
    cos_el = np.cos(inst_el_rad)
    x = - cos_el * np.sin(inst_az_rad)
    y = cos_el * np.cos(inst_az_rad)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def compute_velocity_in_dsp_frame_km_s(energy: np.ndarray, inst_el: np.ndarray, inst_az: np.ndarray) -> np.ndarray:
    particle_direction = - compute_look_directions(inst_el, inst_az)
    speed_meters_per_second = np.sqrt(energy * PROTON_CHARGE_COULOMBS * 2 / ELECTRON_MASS_KG)
    speed_km_per_second = speed_meters_per_second / METERS_PER_KILOMETER

    for i in range(len(particle_direction)):
        particle_direction[i] *= speed_km_per_second[i]

    return particle_direction


def compute_velocity_in_sw_frame(velocity_in_despun_frame, solar_wind_velocity):
    return velocity_in_despun_frame - solar_wind_velocity


def compute_energy_in_ev_from_velocity_in_km_per_second(velocity):
    pass
