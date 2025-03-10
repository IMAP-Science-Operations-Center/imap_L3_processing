import numpy as np
from uncertainties.unumpy import uarray, nominal_values, std_devs


def get_hit_bin_polar_coordinates(declination_bins=8, inclination_bins=15) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declination_starts, declination_step = np.linspace(0, 180, declination_bins, endpoint=False, retstep=True)
    declination_delta = declination_step / 2
    declinations = declination_starts + declination_delta
    inclination_starts, inclination_step = np.linspace(0, 360, inclination_bins, endpoint=False, retstep=True)
    inclination_delta = inclination_step / 2
    inclinations = inclination_starts + inclination_delta
    declination_deltas = np.array([declination_delta] * declination_bins)
    inclination_deltas = np.array([inclination_delta] * inclination_bins)

    return declinations, inclinations, declination_deltas, inclination_deltas


def get_sector_unit_vectors(declinations_degrees: np.ndarray, inclinations_degrees: np.ndarray) -> np.ndarray:
    declinations = np.deg2rad(declinations_degrees)
    inclinations = np.deg2rad(inclinations_degrees)
    declinations = declinations[:, np.newaxis]
    z = np.cos(declinations)
    sin_dec = np.sin(declinations)
    x = sin_dec * np.cos(inclinations)
    y = sin_dec * np.sin(inclinations)
    stacked = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    return stacked


def rebin_by_pitch_angle_and_gyrophase(flux_data: np.array,
                                       flux_delta_plus: np.array,
                                       flux_delta_minus: np.array,
                                       pitch_angles: np.array,
                                       gyrophases: np.array,
                                       number_of_pitch_angle_bins: int,
                                       number_of_gyrophase_bins: int):
    pitch_angle_bins = np.floor(pitch_angles / (180 / number_of_pitch_angle_bins)).astype(int)
    gyrophase_bins = np.floor(gyrophases / (360 / number_of_gyrophase_bins)).astype(int)

    flux_with_delta_plus = uarray(flux_data, flux_delta_plus)
    flux_with_delta_minus = uarray(flux_data, flux_delta_minus)

    output_shape = (flux_data.shape[0], number_of_pitch_angle_bins, number_of_gyrophase_bins)
    rebinned_summed_with_delta_plus = uarray(np.zeros(shape=output_shape), 0)
    rebinned_summed_with_delta_minus = uarray(np.zeros(shape=output_shape), 0)
    rebinned_count = np.zeros(shape=output_shape)

    for i, (flux_delta_plus, flux_delta_minus) in enumerate(zip(flux_with_delta_plus, flux_with_delta_minus)):
        for pitch_angle_bin, gyrophase_bin, flux_with_plus, flux_with_minus in zip(np.ravel(pitch_angle_bins),
                                                                                   np.ravel(gyrophase_bins),
                                                                                   np.ravel(flux_delta_plus),
                                                                                   np.ravel(flux_delta_minus)):
            rebinned_summed_with_delta_plus[i, pitch_angle_bin, gyrophase_bin] += flux_with_plus
            rebinned_summed_with_delta_minus[i, pitch_angle_bin, gyrophase_bin] += flux_with_minus

            rebinned_count[i, pitch_angle_bin, gyrophase_bin] += 1

    averaged_rebinned_fluxes_with_delta_plus = np.divide(rebinned_summed_with_delta_plus, rebinned_count,
                                                         out=np.full_like(rebinned_summed_with_delta_plus, np.nan),
                                                         where=rebinned_count != 0)
    averaged_rebinned_fluxes_with_delta_minus = np.divide(rebinned_summed_with_delta_minus, rebinned_count,
                                                          out=np.full_like(rebinned_summed_with_delta_minus, np.nan),
                                                          where=rebinned_count != 0)
    pa_only_with_delta_plus = np.nanmean(averaged_rebinned_fluxes_with_delta_plus, axis=-1)
    pa_only_with_delta_minus = np.nanmean(averaged_rebinned_fluxes_with_delta_minus, axis=-1)

    return (nominal_values(averaged_rebinned_fluxes_with_delta_plus),
            std_devs(averaged_rebinned_fluxes_with_delta_plus),
            std_devs(averaged_rebinned_fluxes_with_delta_minus),
            nominal_values(pa_only_with_delta_plus),
            std_devs(pa_only_with_delta_plus),
            std_devs(pa_only_with_delta_minus))
