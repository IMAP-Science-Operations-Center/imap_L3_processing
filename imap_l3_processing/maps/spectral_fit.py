import dataclasses

import numpy as np

from imap_l3_processing.maps.map_models import SpectralIndexMapData, IntensityMapData, \
    calculate_datetime_weighted_average
from imap_l3_processing.maps.mpfit import mpfit


def calculate_spectral_index_for_multiple_ranges(intensity_data: IntensityMapData, map_ranges) -> SpectralIndexMapData:
    spectral_maps = []
    for start, end in map_ranges:
        spectral_maps.append(
            fit_spectral_index_map(
                slice_energy_range(intensity_data, start, end)
            )
        )
    return SpectralIndexMapData(
        epoch=intensity_data.epoch,
        epoch_delta=intensity_data.epoch_delta,
        energy=np.concat([m.energy for m in spectral_maps]),
        energy_delta_plus=np.concat([m.energy_delta_plus for m in spectral_maps]),
        energy_delta_minus=np.concat([m.energy_delta_minus for m in spectral_maps]),
        energy_label=np.concat([m.energy_label for m in spectral_maps]),
        latitude=intensity_data.latitude,
        longitude=intensity_data.longitude,
        exposure_factor=np.concat([m.exposure_factor for m in spectral_maps], axis=1),
        obs_date=np.concat([m.obs_date for m in spectral_maps], axis=1),
        obs_date_range=np.concat([m.obs_date_range for m in spectral_maps], axis=1),
        solid_angle=intensity_data.solid_angle,
        ena_spectral_index=np.concat([m.ena_spectral_index for m in spectral_maps], axis=1),
        ena_spectral_index_stat_unc=np.concat([m.ena_spectral_index_stat_unc for m in spectral_maps], axis=1),
    )


def slice_energy_range(data: IntensityMapData, start: float, end: float) -> IntensityMapData:
    energy_mask = np.logical_and(data.energy >= start, data.energy < end)
    return dataclasses.replace(data,
                               energy=data.energy[energy_mask],
                               energy_delta_plus=data.energy_delta_plus[energy_mask],
                               energy_delta_minus=data.energy_delta_minus[energy_mask],
                               energy_label=data.energy_label[energy_mask],
                               exposure_factor=data.exposure_factor[:, energy_mask],
                               obs_date=data.obs_date[:, energy_mask],
                               obs_date_range=data.obs_date_range[:, energy_mask],
                               ena_intensity=data.ena_intensity[:, energy_mask],
                               ena_intensity_sys_err=data.ena_intensity_sys_err[:, energy_mask],
                               ena_intensity_stat_unc=data.ena_intensity_stat_unc[:, energy_mask]
                               )


def fit_spectral_index_map(intensity_data: IntensityMapData) -> SpectralIndexMapData:
    fluxes = intensity_data.ena_intensity
    uncertainty = intensity_data.ena_intensity_stat_unc
    energy = intensity_data.energy

    min_energy = intensity_data.energy[0] - intensity_data.energy_delta_minus[0]
    max_energy = intensity_data.energy[-1] + intensity_data.energy_delta_plus[-1]
    mean_energy = np.sqrt(min_energy * max_energy)
    new_energy_label = f"{min_energy} - {max_energy}"

    output_gammas, output_gamma_errors = fit_arrays_to_power_law(fluxes, uncertainty, energy)
    mean_obs_date = calculate_datetime_weighted_average(intensity_data.obs_date,
                                                        weights=intensity_data.exposure_factor,
                                                        axis=1, keepdims=True)
    mean_obs_date_range = np.ma.average(intensity_data.obs_date_range, weights=intensity_data.exposure_factor,
                                        axis=1,
                                        keepdims=True)
    total_exposure_factor = np.sum(intensity_data.exposure_factor, axis=1, keepdims=True)

    return SpectralIndexMapData(
        epoch=intensity_data.epoch,
        epoch_delta=intensity_data.epoch_delta,
        energy=np.array([mean_energy]),
        energy_delta_plus=np.array([max_energy - mean_energy]),
        energy_delta_minus=np.array([mean_energy - min_energy]),
        energy_label=np.array([new_energy_label]),
        latitude=intensity_data.latitude,
        longitude=intensity_data.longitude,
        exposure_factor=total_exposure_factor,
        obs_date=mean_obs_date,
        obs_date_range=mean_obs_date_range,
        solid_angle=intensity_data.solid_angle,
        ena_spectral_index=output_gammas,
        ena_spectral_index_stat_unc=output_gamma_errors,
    )


def fit_arrays_to_power_law(fluxes: np.ndarray, uncertainties: np.ndarray, energy: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    par_info = [
        {'limits': [0.0, 1000.0]},
        {'limits': [0.0, 1000.0]},
    ]

    output_shape = (fluxes.shape[0], 1, *fluxes.shape[2:])
    output_gammas = np.full(output_shape, np.nan, dtype=float)
    output_gamma_errors = np.full_like(output_gammas, np.nan)

    for epoch in range(fluxes.shape[0]):
        intensity = fluxes[epoch].reshape((fluxes[epoch].shape[0], -1))
        unc = uncertainties[epoch].reshape((uncertainties[epoch].shape[0], -1))

        gammas = np.full(intensity.shape[-1], np.nan, dtype=float)
        errors = np.full_like(gammas, np.nan)
        for i in range(intensity.shape[-1]):
            flux = intensity[:, i]
            uncertainty = unc[:, i]
            flux_and_variance_are_zero = np.equal(flux, 0) & np.equal(uncertainty, 0)
            flux_or_error_is_invalid = np.isnan(flux) | np.isnan(uncertainty) | flux_and_variance_are_zero
            flux = flux[~flux_or_error_is_invalid]
            uncertainty = uncertainty[~flux_or_error_is_invalid]
            filtered_energy = energy[~flux_or_error_is_invalid]

            if len(flux) > 1:
                keywords = {'xval': filtered_energy, 'yval': flux, 'errval': uncertainty}
                first_y_in_range, last_y_in_range = np.log10(flux[flux > 0][0]), np.log10(flux[flux > 0][-1])
                first_x_in_range, last_x_in_range = np.log10(filtered_energy[0]), np.log10(filtered_energy[-1])
                initial_gamma = (last_y_in_range - first_y_in_range) / (last_x_in_range - first_x_in_range)
                initial_intercept = first_y_in_range - (initial_gamma * first_x_in_range)
                initial_parameters = (initial_intercept, -initial_gamma)

                fit = mpfit(power_law, initial_parameters, keywords, par_info, nprint=0)

                a, gamma = fit.params
                if fit.status > 0:
                    a_error, gamma_error = fit.perror
                    gammas[i] = gamma
                    errors[i] = gamma_error
        output_gammas[epoch, 0] = gammas.reshape(fluxes.shape[2:])
        output_gamma_errors[epoch, 0] = errors.reshape(fluxes.shape[2:])
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
