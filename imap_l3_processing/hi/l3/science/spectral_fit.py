import numpy as np

from imap_l3_processing.hi.l3.science.mpfit import mpfit


def spectral_fit(num_epochs, num_lons, num_lats, fluxes, variances, energy):
    initial_parameters = (1, 1)

    gammas = np.full((num_epochs, num_lons, num_lats), fill_value=np.nan, dtype=float)
    for epoch in range(num_epochs):
        for lon in range(num_lons):
            for lat in range(num_lats):
                flux = fluxes[epoch][lon][lat]
                variance = variances[epoch][lon][lat]
                flux_or_error_is_nan = np.isnan(flux) | np.isnan(variance)
                flux = flux[~flux_or_error_is_nan]
                variance = variance[~flux_or_error_is_nan]
                filtered_energy = energy[~flux_or_error_is_nan]
                keywords = {'xval': filtered_energy, 'yval': flux, 'errval': variance}
                _, gamma = mpfit(power_law, initial_parameters, keywords, nprint=0, maxiter=50).params
                gammas[epoch][lon][lat] = gamma
    return gammas


def power_law(params, **kwargs):
    A, B = params
    x = kwargs['xval']
    y = kwargs['yval']
    err = kwargs['errval']

    model = A * np.power(x, -B)

    status = 0
    residuals = (y - model) / err
    # print(f'A: {A}\nB: {B}\nresiduals: {residuals}')

    return status, residuals
