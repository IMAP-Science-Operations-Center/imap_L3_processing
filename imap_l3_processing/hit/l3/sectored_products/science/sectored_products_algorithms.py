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


def get_sector_unit_vectors_codice(elevation: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    elevations = np.deg2rad(elevation)
    azimuths = np.deg2rad(azimuth)
    elevations = elevations[:, np.newaxis]
    z = np.cos(elevations)
    sin_dec = np.sin(elevations)
    x = sin_dec * np.cos(azimuths)
    y = sin_dec * np.sin(azimuths)

    stacked = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    return stacked


def hit_rebin_by_pitch_angle_and_gyrophase(intensity_data: np.array,
                                           intensity_delta_plus: np.array,
                                           intensity_delta_minus: np.array,
                                           pitch_angles: np.array,
                                           gyrophases: np.array,
                                           number_of_pitch_angle_bins: int,
                                           number_of_gyrophase_bins: int):
    intensity_with_delta_plus = uarray(intensity_data, intensity_delta_plus)
    intensity_with_delta_minus = uarray(intensity_data, intensity_delta_minus)

    output_shape_pa_and_gyro = (intensity_data.shape[0], number_of_pitch_angle_bins, number_of_gyrophase_bins)
    output_shape_pa_only = (intensity_data.shape[0], number_of_pitch_angle_bins)

    rebinned_summed_by_pa_and_gyro_with_delta_plus = uarray(np.zeros(shape=output_shape_pa_and_gyro), 0)
    rebinned_summed_by_pa_and_gyro_with_delta_minus = uarray(np.zeros(shape=output_shape_pa_and_gyro), 0)

    rebinned_summed_pa_only_with_delta_plus = uarray(np.zeros(shape=output_shape_pa_only), 0)
    rebinned_summed_pa_only_with_delta_minus = uarray(np.zeros(shape=output_shape_pa_only), 0)

    rebinned_count_by_pa_and_gyro = np.zeros(shape=output_shape_pa_and_gyro)
    rebinned_count_pa_only = np.zeros(shape=output_shape_pa_only)

    for i in range(intensity_data.shape[0]):
        for pitch_angle, gyrophase, intensity_with_plus, intensity_with_minus in zip(
                np.ravel(pitch_angles),
                np.ravel(gyrophases),
                np.ravel(intensity_with_delta_plus[i]),
                np.ravel(intensity_with_delta_minus[i])):
            if not (np.isnan(intensity_with_plus.nominal_value) or np.isnan(pitch_angle)):
                pitch_angle_bin = np.floor(pitch_angle / (180 / number_of_pitch_angle_bins)).astype(int)
                if not np.isnan(gyrophase):
                    gyrophase_bin = np.floor(gyrophase / (360 / number_of_gyrophase_bins)).astype(int)

                    rebinned_summed_by_pa_and_gyro_with_delta_plus[
                        i, pitch_angle_bin, gyrophase_bin] += intensity_with_plus
                    rebinned_summed_by_pa_and_gyro_with_delta_minus[
                        i, pitch_angle_bin, gyrophase_bin] += intensity_with_minus
                    rebinned_count_by_pa_and_gyro[i, pitch_angle_bin, gyrophase_bin] += 1

                rebinned_summed_pa_only_with_delta_plus[i, pitch_angle_bin] += intensity_with_plus
                rebinned_summed_pa_only_with_delta_minus[i, pitch_angle_bin] += intensity_with_minus
                rebinned_count_pa_only[i, pitch_angle_bin] += 1

    averaged_rebinned_intensity_by_pa_and_gyro_with_delta_plus = np.divide(
        rebinned_summed_by_pa_and_gyro_with_delta_plus,
        rebinned_count_by_pa_and_gyro,
        out=np.full_like(
            rebinned_summed_by_pa_and_gyro_with_delta_plus, np.nan),
        where=rebinned_count_by_pa_and_gyro != 0)
    averaged_rebinned_intensity_by_pa_and_gyro_with_delta_minus = np.divide(
        rebinned_summed_by_pa_and_gyro_with_delta_minus,
        rebinned_count_by_pa_and_gyro,
        out=np.full_like(
            rebinned_summed_by_pa_and_gyro_with_delta_minus, np.nan),
        where=rebinned_count_by_pa_and_gyro != 0)

    averaged_rebinned_intensity_by_pa_only_with_delta_plus = np.divide(rebinned_summed_pa_only_with_delta_plus,
                                                                       rebinned_count_pa_only,
                                                                       out=np.full_like(
                                                                           rebinned_summed_pa_only_with_delta_plus,
                                                                           np.nan),
                                                                       where=rebinned_count_pa_only != 0)
    averaged_rebinned_intensity_by_pa_only_with_delta_minus = np.divide(rebinned_summed_pa_only_with_delta_minus,
                                                                        rebinned_count_pa_only,
                                                                        out=np.full_like(
                                                                            rebinned_summed_pa_only_with_delta_minus,
                                                                            np.nan),
                                                                        where=rebinned_count_pa_only != 0)

    return (nominal_values(averaged_rebinned_intensity_by_pa_and_gyro_with_delta_plus),
            std_devs(averaged_rebinned_intensity_by_pa_and_gyro_with_delta_plus),
            std_devs(averaged_rebinned_intensity_by_pa_and_gyro_with_delta_minus),
            nominal_values(averaged_rebinned_intensity_by_pa_only_with_delta_plus),
            std_devs(averaged_rebinned_intensity_by_pa_only_with_delta_plus),
            std_devs(averaged_rebinned_intensity_by_pa_only_with_delta_minus))
