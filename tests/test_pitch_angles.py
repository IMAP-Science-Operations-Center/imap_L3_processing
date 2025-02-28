import unittest

import numpy as np

from imap_processing.pitch_angles import calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase


class TestPitchAngles(unittest.TestCase):

    def test_calculate_pitch_angle(self):
        hit_unit_vector = np.array([-0.09362045, 0.8466484, 0.5238528])
        mag_unit_vector = np.array([-0.42566603, 0.7890057, 0.44303328])

        actual_pitch_angle = calculate_pitch_angle(hit_unit_vector, mag_unit_vector)
        self.assertAlmostEqual(19.95757200714941, actual_pitch_angle)

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

    def test_gyrophase(self):
        particle_vector = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])
        mag_field_vector = np.array([0, 0, 1])

        gyrophases = calculate_gyrophase(particle_vector, mag_field_vector)

        expected_gyrophases = [0, 90, 0, 180, 270, 0]

        np.testing.assert_array_equal(gyrophases, expected_gyrophases)


if __name__ == '__main__':
    unittest.main()
