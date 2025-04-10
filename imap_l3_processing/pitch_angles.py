import numpy as np


def calculate_pitch_angle(particle_vectors: np.ndarray[float], magnetic_field_vector: np.ndarray[float]) -> np.ndarray[
    float]:
    norm_x = calculate_unit_vector(particle_vectors)
    norm_y = calculate_unit_vector(magnetic_field_vector)
    dot = np.sum(norm_x * norm_y, axis=-1)
    return np.degrees(np.acos(dot))


def calculate_unit_vector(vector: np.ndarray[float]) -> np.ndarray[float]:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / norm


def rotate_from_imap_despun_to_hit_despun(vector: np.ndarray[float]) -> np.ndarray[float]:
    rotation_matrix_from_imap_despun_frame_to_hit_despun_instrument_frame = [[0.866025, 0.5, 0],
                                                                             [-0.5, 0.866025, 0],
                                                                             [0, 0, 1]]
    return rotation_matrix_from_imap_despun_frame_to_hit_despun_instrument_frame @ vector


def rotate_particle_vectors_from_hit_despun_to_imap_despun(vector: np.ndarray[float]) -> np.ndarray[float]:
    rotation_matrix = [[0.866025, -0.5, 0],
                       [0.5, 0.866025, 0],
                       [0, 0, 1]]

    output = np.full_like(vector, np.nan, dtype=float)

    for pitch_angle_index, pitch_angle_bin in enumerate(vector):
        for gyrophase_index, gyrophase_bin in enumerate(pitch_angle_bin):
            output[pitch_angle_index][gyrophase_index] = rotation_matrix @ gyrophase_bin

    return output


def calculate_gyrophase(particle_vectors: np.ndarray, magnetic_field_vector: np.ndarray):
    magnetic_field_plus_z = magnetic_field_vector
    imap_dps_plus_y = [0, 1, 0]
    if np.all(np.cross(magnetic_field_plus_z, imap_dps_plus_y) == 0):
        return np.full(shape=particle_vectors.shape[:-1], fill_value=np.nan)
    magnetic_field_plus_x = calculate_unit_vector(np.cross(imap_dps_plus_y, magnetic_field_plus_z))
    magnetic_field_plus_y = calculate_unit_vector(np.cross(magnetic_field_plus_z, magnetic_field_plus_x))
    particle_magnetic_field_x_component = np.sum(particle_vectors * magnetic_field_plus_x, axis=-1)
    particle_magnetic_field_y_component = np.sum(particle_vectors * magnetic_field_plus_y, axis=-1)
    gyrophases = np.atan2(particle_magnetic_field_x_component, particle_magnetic_field_y_component)

    return np.mod(np.degrees(gyrophases), 360)
