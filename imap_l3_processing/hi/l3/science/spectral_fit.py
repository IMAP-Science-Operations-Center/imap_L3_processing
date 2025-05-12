import numpy as np

from imap_l3_processing.hi.l3.science.mpfit import mpfit


def spectral_fit(fluxes, variances, energy, output_energy=None):
    output_energy = output_energy if output_energy is not None else np.array([[-np.inf, np.inf]])
    initial_parameters = (10, 2)

    par_info = [
        {'limits': [0.0, 1000.0]},
        {'limits': [0.0, 1000.0]},
    ]

    output_shape = (fluxes.shape[0], output_energy.shape[0], *fluxes.shape[2:])
    output_gammas = np.full(output_shape, np.nan, dtype=float)
    output_gamma_errors = np.full_like(output_gammas, np.nan)

    for epoch in range(fluxes.shape[0]):
        for output_energy_index in range(output_energy.shape[0]):
            intensity = fluxes[epoch].reshape((fluxes[epoch].shape[0], -1))
            var = variances[epoch].reshape((variances[epoch].shape[0], -1))

            gammas = np.full(intensity.shape[-1], np.nan, dtype=float)
            errors = np.full_like(gammas, np.nan)
            for i in range(intensity.shape[-1]):
                flux = intensity[:, i]
                variance = var[:, i]
                energy_mask = (energy >= output_energy[output_energy_index][0]) & (
                        energy < output_energy[output_energy_index][1])
                flux_and_variance_are_zero = np.equal(flux, 0) & np.equal(variance, 0)
                flux_or_error_is_invalid = np.isnan(flux) | np.isnan(variance) | flux_and_variance_are_zero
                flux = flux[~flux_or_error_is_invalid & energy_mask]
                variance = variance[~flux_or_error_is_invalid & energy_mask]
                filtered_energy = energy[~flux_or_error_is_invalid & energy_mask]
                keywords = {'xval': filtered_energy, 'yval': flux, 'errval': np.sqrt(variance)}
                fit = mpfit(power_law, initial_parameters, keywords, par_info, nprint=0)

                a, gamma = fit.params
                if fit.status > 0:
                    a_error, gamma_error = fit.perror
                    gammas[i] = gamma
                    errors[i] = gamma_error
            output_gammas[epoch, output_energy_index] = gammas.reshape(fluxes.shape[2:])
            output_gamma_errors[epoch, output_energy_index] = errors.reshape(fluxes.shape[2:])

    return output_gammas, output_gamma_errors


def power_law(params, **kwargs):
    A, B = params
    x = kwargs['xval']
    y = kwargs['yval']
    err = kwargs['errval']

    model = A * np.power(x, -B)

    status = 0
    residuals = (y - model) / err

    return status, residuals
