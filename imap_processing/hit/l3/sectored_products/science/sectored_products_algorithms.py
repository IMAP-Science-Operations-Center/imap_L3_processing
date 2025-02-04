import numpy as np


def get_sector_unit_vectors() -> np.ndarray:
    starts, step = np.linspace(0, 180, 8, endpoint=False, retstep=True)
    declinations = starts + step / 2
    starts, step = np.linspace(0, 360, 15, endpoint=False, retstep=True)
    azimuths = starts + step / 2

    offset = 0
    declinations_rad = np.deg2rad(declinations)[:, np.newaxis]
    azimuths_rad = np.deg2rad(azimuths + offset)
    z = np.cos(declinations_rad)
    sin_dec = np.sin(declinations_rad)
    x = sin_dec * np.cos(azimuths_rad)
    y = sin_dec * np.sin(azimuths_rad)
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
