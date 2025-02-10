from datetime import timedelta

import numpy as np


def calculate_average_mag_data(mag_data: np.ndarray, mag_epoch: np.ndarray, hit_epoch: np.ndarray,
                               hit_delta: np.ndarray) -> np.ndarray:
    vector_sums = np.zeros(shape=(hit_epoch.shape[0], 3), dtype=float)
    vector_counts = np.zeros_like(hit_epoch, dtype=float)

    mag_data_iter = ((time, vec) for time, vec in zip(mag_epoch, mag_data))
    current_epoch, current_vec = next(mag_data_iter)
    for i, (time, delta) in enumerate(zip(hit_epoch, hit_delta)):
        start_time = time - delta
        end_time = time + delta

        while current_epoch < start_time:
            current_epoch, current_vec = next(mag_data_iter)

        while current_epoch is not None and start_time <= current_epoch < end_time:
            vector_sums[i] += current_vec
            vector_counts[i] += 1
            current_epoch, current_vec = next(mag_data_iter, (None, None))

    vector_counts = np.reshape(vector_counts, (-1, 1))
    return np.divide(vector_sums, vector_counts, out=np.full_like(vector_sums, fill_value=np.nan), where=vector_counts != 0)


def get_hit_bin_polar_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declination_starts, declination_step = np.linspace(0, 180, 8, endpoint=False, retstep=True)
    declination_delta = declination_step / 2
    declinations = declination_starts + declination_delta
    azimuth_starts, azimuth_step = np.linspace(0, 360, 15, endpoint=False, retstep=True)
    azimuth_delta = azimuth_step / 2
    azimuths = azimuth_starts + azimuth_delta
    return declinations, azimuths, declination_delta, azimuth_delta


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


def calculate_pitch_angle(x: np.ndarray[float], y: np.ndarray[float]) -> float:
    norm_x = calculate_unit_vector(x)
    norm_y = calculate_unit_vector(y)
    return np.degrees(np.acos(np.dot(norm_x, norm_y)))


def calculate_unit_vector(vector: np.ndarray[float]) -> np.ndarray[float]:
    return (vector.T / np.linalg.norm(vector, axis=-1)).T


def calculate_gyrophase(particle_vectors: np.ndarray, magnetic_field_vector: np.ndarray):
    magnetic_field_plus_z = magnetic_field_vector
    imap_dps_plus_x = [1, 0, 0]
    magnetic_field_plus_y = calculate_unit_vector(np.cross(magnetic_field_plus_z, imap_dps_plus_x))
    magnetic_field_plus_x = calculate_unit_vector(np.cross(magnetic_field_plus_y, magnetic_field_plus_z))
    particle_magnetic_field_x_component = np.dot(particle_vectors, magnetic_field_plus_x)
    particle_magnetic_field_y_component = np.dot(particle_vectors, magnetic_field_plus_y)
    gyrophases = np.atan2(particle_magnetic_field_y_component, particle_magnetic_field_x_component)

    return np.mod(np.degrees(gyrophases), 360)


def calculate_sector_areas(declinations_degrees: np.ndarray, declination_deltas_degrees: np.ndarray,
                           inclination_deltas_degrees: np.ndarray):
    northernmost_angles = np.deg2rad(declinations_degrees) - np.deg2rad(declination_deltas_degrees)
    southernmost_angles = np.deg2rad(declinations_degrees) + np.deg2rad(declination_deltas_degrees)

    declination_term = (np.cos(northernmost_angles) - np.cos(southernmost_angles)).reshape((-1, 1))
    inclination_term = (2 * np.deg2rad(inclination_deltas_degrees)).reshape((1, -1))

    return declination_term * inclination_term


def rebin_by_pitch_angle_and_gyrophase(flux_data: np.array, pitch_angles: np.array, gyrophases: np.array,
                                       sector_areas: np.array,
                                       number_of_pitch_angle_bins: int, number_of_gyrophase_bins: int):
    pitch_angle_bins = np.floor(pitch_angles / (180 / number_of_pitch_angle_bins)).astype(int)
    gyrophase_bins = np.floor(gyrophases / (360 / number_of_gyrophase_bins)).astype(int)

    output_shape = (flux_data.shape[0], number_of_pitch_angle_bins, number_of_gyrophase_bins)
    rebinned_summed = np.zeros(shape=output_shape)
    rebinned_acc = np.zeros(shape=output_shape)

    for i, flux_data in enumerate(flux_data):
        for pitch_angle_bin, gyrophase_bin, sector_area, flux in zip(np.ravel(pitch_angle_bins),
                                                                     np.ravel(gyrophase_bins), np.ravel(sector_areas),
                                                                     np.ravel(flux_data)):
            rebinned_summed[i, pitch_angle_bin, gyrophase_bin] += sector_area * flux
            rebinned_acc[i, pitch_angle_bin, gyrophase_bin] += sector_area
    return np.divide(rebinned_summed, rebinned_acc, out=np.full_like(rebinned_summed, np.nan),
                     where=rebinned_acc != 0)
