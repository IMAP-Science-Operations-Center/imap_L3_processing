import unittest

import numpy as np

from imap_l3_processing.pitch_angles import calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase, \
    rotate_from_imap_despun_to_hit_despun, rotate_particle_vectors_from_hit_despun_to_imap_despun


class TestPitchAngles(unittest.TestCase):

    def test_calculate_pitch_angle(self):
        hit_unit_vector = np.array([[-0.09362045, 0.8466484, 0.5238528]])
        mag_unit_vector = np.array([-0.42566603, 0.7890057, 0.44303328])

        actual_pitch_angle = calculate_pitch_angle(hit_unit_vector, mag_unit_vector)
        expected_pitch_angle = np.array([19.95757200714941])
        self.assertEqual(expected_pitch_angle.shape, actual_pitch_angle.shape)
        np.testing.assert_array_almost_equal(actual_pitch_angle, expected_pitch_angle)

    def test_calculate_pitch_angle_for_multiple_vectors(self):
        hit_unit_vectors = np.array([[1, 1, 0], [1, 0, 0]])
        mag_unit_vector = np.array([1, 0, 0])

        actual_pitch_angle = calculate_pitch_angle(hit_unit_vectors, mag_unit_vector)
        np.testing.assert_array_almost_equal(actual_pitch_angle, [45, 0])

    def test_calculate_pitch_angle_for_multiple_mag_vectors(self):
        hit_unit_vectors = np.array([[1, 1, 0], [1, 0, 0]])
        mag_unit_vector = np.array([[1, 0, 0], [0, 0, 1]])
        actual_pitch_angle = calculate_pitch_angle(hit_unit_vectors, mag_unit_vector)
        np.testing.assert_array_almost_equal(actual_pitch_angle, [45, 90])

    def test_calculate_unit_vector(self):
        vector = np.array([27, 34, 56])

        unit_vector = calculate_unit_vector(vector)

        np.testing.assert_array_almost_equal(np.array([0.38103832, 0.47982603, 0.7903017]), unit_vector)

        many_vectors = np.array([[0.1, 0, 0], [30, 40, 0]])
        unit_vectors = calculate_unit_vector(many_vectors)
        np.testing.assert_array_almost_equal([[1, 0, 0], [3 / 5, 4 / 5, 0]], unit_vectors)

    def test_rotate_from_imap_despun_to_hit_despun(self):
        vector = np.array([1.0, 1.0, 1.0])

        rotated_vector = rotate_from_imap_despun_to_hit_despun(vector)

        expected_vector = np.array([1.366025, 0.366025, 1.0])

        np.testing.assert_array_almost_equal(rotated_vector, expected_vector)

    def test_rotate_particle_vectors_from_hit_despun_to_imap_despun(self):
        particle_vectors = np.array([
            [[0, 0, 1], [1, 0, 0]],
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [1, 0, 1]],
            [[0, 0, 0], [1, 1, 1]]
        ], dtype=float)

        rotated_particle_vectors = rotate_particle_vectors_from_hit_despun_to_imap_despun(particle_vectors)

        expected_particle_vectors = np.array([
            [[0, 0, 1], [0.866025, 0.5, 0.]],
            [[0.366025, 1.366025, 0.], [-0.5, 0.866025, 1.]],
            [[- 0.5, 0.866025, 0.], [0.866025, 0.5, 1.]],
            [[0, 0, 0], [0.366025, 1.366025, 1.]]
        ])

        np.testing.assert_array_almost_equal(rotated_particle_vectors, expected_particle_vectors)

    def test_gyrophase(self):
        particle_vector = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [.5, .5, 0],
            [.5, -.5, 0],
            [0, 0, -1],
            [0, 0, 1],
        ])
        mag_field_vector = np.array([0, 0, 1])

        gyrophases = calculate_gyrophase(particle_vector, mag_field_vector)

        expected_gyrophases = [0, 90, 180, 270, 45, 135, 0, 0]

        np.testing.assert_array_equal(gyrophases, expected_gyrophases)

    def test_gyrophase_degenerate_particle_and_magnetic_field_returns_zero(self):
        particle_vector = np.array([
            [0, 0, 1],
        ])
        mag_field_vector = np.array([0, 0, 1])

        gyrophases = calculate_gyrophase(particle_vector, mag_field_vector)

        expected_gyrophases = [0]

        np.testing.assert_array_equal(gyrophases, expected_gyrophases)

    def test_gyrophase_degenerate_dps_frame_y_axis_and_magnetic_field_returns_all_nans(self):
        particle_vector = np.array([[
            [1, 0, 0],
            [-1, 0, 0],
        ]])

        for mag_field_vector in [np.array([0, 1, 0]), np.array([0, -1, 0])]:
            with self.subTest(mag_field_vector):
                gyrophases = calculate_gyrophase(particle_vector, mag_field_vector)

                expected_gyrophases = [[np.nan, np.nan]]
                np.testing.assert_array_equal(gyrophases, expected_gyrophases)


if __name__ == '__main__':
    unittest.main()
