import numpy as np


def get_hit_bin_polar_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    declination_starts, declination_step = np.linspace(0, 180, 8, endpoint=False, retstep=True)
    declination_delta = declination_step / 2
    declinations = declination_starts + declination_delta
    azimuth_starts, azimuth_step = np.linspace(0, 360, 15, endpoint=False, retstep=True)
    azimuth_delta = azimuth_step / 2
    azimuths = azimuth_starts + azimuth_delta
    return np.deg2rad(declinations), np.deg2rad(azimuths), np.deg2rad(declination_delta), np.deg2rad(azimuth_delta)


def get_sector_unit_vectors() -> np.ndarray:
    declinations, azimuths, _, _ = get_hit_bin_polar_coordinates()
    declinations = declinations[:, np.newaxis]
    z = np.cos(declinations)
    sin_dec = np.sin(declinations)
    x = sin_dec * np.cos(azimuths)
    y = sin_dec * np.sin(azimuths)
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


def rebin_by_pitch_angle_and_gyrophase(mag_coordinate_system: np.ndarray, flux_data: np.ndarray):
    pitch_angle_bins = get_sector_unit_vectors()
    flattened_pitch_angle_vectors = pitch_angle_bins.reshape(-1, pitch_angle_bins.shape[-1])
    output_fluxes = []
    input_sector_vectors = np.dot(mag_coordinate_system, flattened_pitch_angle_vectors.T).T

    flux_for_first_energy = flux_data[0]
    flattened_flux_for_first_energy = flux_for_first_energy.reshape(-1, flux_for_first_energy.shape[-1])
    for i, pitch_patch in enumerate(pitch_angle_bins):
        for input_patch in input_sector_vectors:
            vector_equality_mask = pitch_patch == input_patch)
            if np.all(vector_equality_mask):
                output_fluxes.append(flattened_flux_for_first_energy[i])

    output_array = np.array(output_fluxes).reshape(8, 15)
    return output_array
