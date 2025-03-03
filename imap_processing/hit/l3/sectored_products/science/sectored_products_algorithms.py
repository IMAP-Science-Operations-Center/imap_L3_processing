import numpy as np


def get_hit_bin_polar_coordinates(declination_bins=8, inclination_bins=15) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declination_starts, declination_step = np.linspace(0, 180, declination_bins, endpoint=False, retstep=True)
    declination_delta = declination_step / 2
    declinations = declination_starts + declination_delta
    inclination_starts, inclination_step = np.linspace(0, 360, inclination_bins, endpoint=False, retstep=True)
    inclination_delta = inclination_step / 2
    inclinations = inclination_starts + inclination_delta
    return declinations, inclinations, declination_delta, inclination_delta


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

    output_shape = (flux_data.shape[0], number_of_pitch_angle_bins, number_of_gyrophase_bins)
    rebinned_summed = np.zeros(shape=output_shape)
    rebinned_count = np.zeros(shape=output_shape)

    for i, flux_data in enumerate(flux_data):
        for pitch_angle_bin, gyrophase_bin, flux in zip(np.ravel(pitch_angle_bins),
                                                        np.ravel(gyrophase_bins),
                                                        np.ravel(flux_data)):
            rebinned_summed[i, pitch_angle_bin, gyrophase_bin] += flux

            rebinned_count[i, pitch_angle_bin, gyrophase_bin] += 1

    averaged_rebinned_fluxes = np.divide(rebinned_summed, rebinned_count, out=np.full_like(rebinned_summed, np.nan),
                                         where=rebinned_count != 0)
    return averaged_rebinned_fluxes, None, None, np.nansum(averaged_rebinned_fluxes, axis=-1), None, None
