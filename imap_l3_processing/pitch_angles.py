import numpy as np
from uncertainties.unumpy import nominal_values, std_devs, uarray


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


def rebin_by_pitch_angle_and_gyrophase(intensity_data: np.array,
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
