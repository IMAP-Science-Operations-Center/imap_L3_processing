import numpy as np
from scipy.optimize import curve_fit


def piece_wise_model(x, b0, b1, b2, b3, b4, b5):
    return np.log(np.piecewise(x, [x <= b2, (x > b2) & (x <= b4), x > b4],
                               [
                                   lambda x: b0 * np.exp(-b1 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(-b3 * x),
                                   lambda x: b0 * np.exp(b2 * (b3 - b1)) * np.exp(b4 * (b5 - b3)) * np.exp(-b5 * x),
                               ]))


def find_breakpoints(energies, flux):
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

    fit, _ = curve_fit(piece_wise_model, energies, log_flux, initial_guesses)
    return fit[2], fit[4]
