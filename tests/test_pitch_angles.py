import unittest

import numpy as np

from imap_l3_processing.pitch_angles import calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase, \
    rotate_from_imap_despun_to_hit_despun, rotate_particle_vectors_from_hit_despun_to_imap_despun, \
    rebin_by_pitch_angle_and_gyrophase


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

    def test_calculate_gyrophase(self):
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

    def test_calculate_gyrophase_fany_input_arrays(self):
        particle_vector = np.array([
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, -1, 0],
                [-1, 0, 0]
            ],
            [
                [.5, .5, 0],
                [.5, -.5, 0],
                [0, 0, -1],
                [0, 0, 1],
            ]
        ])
        mag_field_vector = np.array([[0, 0, 1], [0, 0, 1]])[..., np.newaxis, :]

        gyrophases = calculate_gyrophase(particle_vector, mag_field_vector)

        expected_gyrophases = [[0, 90, 180, 270], [45, 135, 0, 0]]

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

    def test_rebin_intensity_by_pitch_angle_and_gyrophase(self):
        default_pitch_angles = np.array([[45, 45, 45, 45], [135, 135, 135, 135]])
        default_gyrophases = np.array([[45, 135, 225, 315], [45, 135, 225, 315]])
        intensity_data_for_energy_1 = [[0, 1, 2, 3], [4, 5, 6, 7]]
        intensity_data_for_energy_2 = [[8, 9, 10, 11], [12, 13, 14, 15]]
        intensity = np.array([
            intensity_data_for_energy_1,
            intensity_data_for_energy_2
        ])
        intensity_delta_plus = intensity * 0.09
        intensity_delta_minus = intensity * 0.11

        test_cases = [
            ('One bin', default_pitch_angles, default_gyrophases, 1, 1, [[[3.5]], [[11.5]]], [[3.5], [11.5]]),
            ('Rebinned to same bins', default_pitch_angles, default_gyrophases, 2, 4, intensity,
             np.array([[1.5, 5.5], [9.5, 13.5]])),
            ('Gyrophase rotated by 90', default_pitch_angles, (default_gyrophases + 90) % 360, 2, 4,
             [
                 [[3, 0, 1, 2], [7, 4, 5, 6]],
                 [[11, 8, 9, 10], [15, 12, 13, 14]]
             ],
             np.array([[1.5, 5.5], [9.5, 13.5]])),
            ('Includes empty bins', default_pitch_angles, default_gyrophases, 2, 6,
             [
                 [[0, np.nan, 1, 2, np.nan, 3], [4, np.nan, 5, 6, np.nan, 7]],
                 [[8, np.nan, 9, 10, np.nan, 11], [12, np.nan, 13, 14, np.nan, 15]]
             ],
             np.array([[1.5, 5.5], [9.5, 13.5]])),
            ('Varying Pitch Angles', np.array([[45, 45, 135, 135], [45, 45, 135, 135]]),
             np.array([[135, 225, 225, 135], [45, 315, 315, 45]]), 2, 4,
             [
                 [[4, 0, 1, 5], [7, 3, 2, 6]],
                 [[12, 8, 9, 13], [15, 11, 10, 14]]
             ],
             [[2.5, 4.5], [10.5, 12.5]]),
            ('Nan in pitch angles', np.array([[45, 45, 135, 135], [np.nan, 45, 135, 135]]),
             np.array([[135, 225, 225, 135], [45, 315, 315, 45]]), 2, 4,
             [
                 [[np.nan, 0, 1, 5], [7, 3, 2, 6]],
                 [[np.nan, 8, 9, 13], [15, 11, 10, 14]]
             ],
             [[2, 4.5], [10, 12.5]]),
            (
                'Nan for all pitch angles in a bin', np.array([[np.nan, np.nan, 135, 135], [np.nan, np.nan, 135, 135]]),
                np.array([[135, 225, 225, 135], [45, 315, 315, 45]]), 2, 4,
                [
                    [[np.nan, np.nan, np.nan, np.nan], [7, 3, 2, 6]],
                    [[np.nan, np.nan, np.nan, np.nan], [15, 11, 10, 14]]
                ],
                [[np.nan, 4.5], [np.nan, 12.5]]),
            ('Nan in gyrophase', np.array([[45, 45, 135, 135], [45, 45, 135, 135]]),
             np.array([[np.nan, 225, 225, 135], [45, 315, 315, 45]]), 2, 4,
             [
                 [[4, np.nan, 1, 5], [7, 3, 2, 6]],
                 [[12, np.nan, 9, 13], [15, 11, 10, 14]]
             ],
             [[2.5, 4.5], [10.5, 12.5]]),

            ('Gyrophase bin includes 360 bin', default_pitch_angles,
             np.array([[360, 90, 180, 270], [360, 90, 180, 270]]),
             2, 4, intensity,
             np.array([[1.5, 5.5], [9.5, 13.5]]))
        ]

        for case, pitch_angles, gyrophases, number_of_pitch_angle_bins, number_of_gyrophase_bins, \
                expected_intensity_by_pa_gyrophase, expected_intensity_by_pa in test_cases:
            with (((self.subTest(case)))):
                rebinned_data = rebin_by_pitch_angle_and_gyrophase(intensity, intensity_delta_plus,
                                                                   intensity_delta_minus, pitch_angles, gyrophases,
                                                                   number_of_pitch_angle_bins, number_of_gyrophase_bins)
                intensity_pa_and_gyro, _, _ = rebinned_data[0:3]
                intensity_pa_only, _, _ = rebinned_data[3:6]

                np.testing.assert_array_equal(intensity_pa_and_gyro, expected_intensity_by_pa_gyrophase)
                np.testing.assert_array_equal(intensity_pa_only, expected_intensity_by_pa)

    def test_rebin_by_pitch_angle_and_gyrophase_pa_product_only_correctly_weights_when_averaging(self):
        intensity = np.array([[[0, 1, 2], [3, 4, 5]]])
        rebinned_data = rebin_by_pitch_angle_and_gyrophase(intensity, intensity * 0.01, intensity * 0.01,
                                                           np.array([[45, 45, 45], [135, 135, 135]]),
                                                           np.array([[45, 135, 225], [45, 135, 225]]), 1, 2)
        intensity_pa_and_gyro, _, _ = rebinned_data[0:3]
        intensity_pa_only, _, _ = rebinned_data[3:6]

        np.testing.assert_equal(intensity_pa_and_gyro, [[[2, 3.5]]])
        np.testing.assert_equal(intensity_pa_only, [[2.5]])

    def test_rebin_by_pitch_angle_and_gyrophase_pa_product_only_handles_nan_intensity(self):
        intensity = np.array([[[0, np.nan, 2, 7], [3, 4, np.nan, 8]]])
        rebinned_data = rebin_by_pitch_angle_and_gyrophase(intensity, intensity * 0.01, intensity * 0.01,
                                                           np.array([[45, 45, 45, 45], [135, 135, 135, 135]]),
                                                           np.array([[45, 135, 225, 300], [45, 135, 225, 300]]), 2, 2)
        intensity_pa_and_gyro, _, _ = rebinned_data[0:3]
        intensity_pa_only, _, _ = rebinned_data[3:6]

        np.testing.assert_equal(intensity_pa_and_gyro, [[[0, 4.5], [3.5, 8]]])
        np.testing.assert_equal(intensity_pa_only, [[3, 5]])

    def test_rebin_by_pitch_angle_and_gyrophase_includes_uncertainties(self):
        default_pitch_angles = np.array([[45, 45, 45, 45], [135, 135, 135, 135]])
        default_gyrophases = np.array([[45, 135, 225, 315], [45, 135, 225, 315]])
        intensity_data_for_energy_1 = [[0, 1, 2, 3], [4, 5, 6, 7]]
        intensity_data_for_energy_2 = [[8, 9, 10, 11], [12, 13, 14, 15]]
        intensity = np.array([
            intensity_data_for_energy_1,
            intensity_data_for_energy_2
        ])
        intensity_delta_plus = intensity * 0.09
        intensity_delta_minus = intensity * 0.11

        number_of_pitch_angle_bins = 2
        number_of_gyrophase_bins = 4

        expected_delta_plus_pa_gyro = np.array([
            [[0, 0.09, 0.18, 0.27], [0.36, 0.45, 0.54, 0.63]],
            [[0.72, 0.81, 0.9, 0.99], [1.08, 1.17, 1.26, 1.35]]
        ])

        expected_delta_minus_pa_gyro = np.array([
            [[0, 0.11, 0.22, 0.33], [0.44, 0.55, 0.66, 0.77]],
            [[0.88, 0.99, 1.1, 1.21], [1.32, 1.43, 1.54, 1.65]]
        ])

        expected_delta_plus_pa_only = np.array([
            [0.084187, 0.252562],
            [0.43045, 0.60958]
        ])

        expected_delta_minus_pa_only = np.array([
            [0.102896, 0.308687],
            [0.526106, 0.745042]
        ])

        rebinned_data = rebin_by_pitch_angle_and_gyrophase(intensity, intensity_delta_plus, intensity_delta_minus,
                                                           default_pitch_angles, default_gyrophases,
                                                           number_of_pitch_angle_bins, number_of_gyrophase_bins)

        _, delta_plus_pa_gyro, delta_minus_pa_gyro = rebinned_data[0:3]
        _, delta_plus_pa_only, delta_minus_pa_only = rebinned_data[3:6]

        np.testing.assert_allclose(delta_plus_pa_gyro, expected_delta_plus_pa_gyro)
        np.testing.assert_allclose(delta_minus_pa_gyro, expected_delta_minus_pa_gyro)
        np.testing.assert_allclose(delta_plus_pa_only, expected_delta_plus_pa_only, rtol=1e-5)
        np.testing.assert_allclose(delta_minus_pa_only, expected_delta_minus_pa_only, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
