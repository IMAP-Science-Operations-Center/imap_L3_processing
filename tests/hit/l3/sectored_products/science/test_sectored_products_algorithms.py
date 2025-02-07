from unittest import TestCase

import numpy as np

from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    calculate_pitch_angle, calculate_unit_vector, calculate_gyrophase, get_hit_bin_polar_coordinates, \
    rebin_by_pitch_angle_and_gyrophase


class TestSectoredProductsAlgorithms(TestCase):
    def test_get_sector_unit_vectors(self):
        sector_vectors = get_sector_unit_vectors()
        self.assertEqual((8, 15, 3), sector_vectors.shape)
        np.testing.assert_array_almost_equal([1.90827130e-01, 4.05615587e-02, 9.80785280e-01], sector_vectors[0][0])
        np.testing.assert_array_almost_equal([1.90827130e-01, -4.05615587e-02, -9.80785280e-01], sector_vectors[-1][-1])

    def test_get_hit_bin_polar_coordinates(self):
        declinations, azimuths, declination_delta, azimuth_delta = get_hit_bin_polar_coordinates()

        self.assertEqual(np.deg2rad(11.25), declination_delta)
        self.assertEqual(np.deg2rad(12), azimuth_delta)
        np.testing.assert_array_almost_equal(np.deg2rad(11.25 + np.arange(0, 8) * 22.5), declinations)
        np.testing.assert_array_almost_equal(np.deg2rad(12 + np.arange(15) * 24), azimuths)

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

    def test_calculate_unit_vector(self):
        vector = np.array([27, 34, 56])

        unit_vector = calculate_unit_vector(vector)

        np.testing.assert_array_almost_equal(np.array([0.38103832, 0.47982603, 0.7903017]), unit_vector)

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

    def test_rebin_by_pitch_angle_and_gyrophase(self):
        flux_data_for_energy_1 = np.arange(0, 2 * 4).reshape((2, 4))
        flux_data_for_energy_2 = np.arange(2 * 4, 2 * 2 * 4).reshape((2, 4))
        fluxes = np.array([
            flux_data_for_energy_1,
            flux_data_for_energy_2
        ])
        sector_areas = np.full((2, 4), np.pi / 2)

        """
        [[0, 1, 2, 3], [4, 5, 6, 7],
        [8, 9, 10, 11], [12, 13, 14, 15]]
        """

        pitch_angles = np.array([[45, 45, 45, 45], [135, 135, 135, 135]])
        gyrophases = np.array([[45, 135, 225, 315], [45, 135, 225, 315]])

        fluxes_rebinned_to_one_bin = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles, gyrophases, sector_areas,
                                                                        1, 1)
        np.testing.assert_equal(fluxes_rebinned_to_one_bin, [[[3.5]], [[11.5]]])
        fluxes_rebinned_to_same_bins = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles, gyrophases,
                                                                          sector_areas, 2, 4)
        np.testing.assert_array_almost_equal(fluxes_rebinned_to_same_bins, fluxes)

        fluxes_with_gyrophase_rotated_by_90 = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                                 (gyrophases + 90) % 360,
                                                                                 sector_areas,
                                                                                 2, 4)
        np.testing.assert_array_almost_equal([[3, 0, 1, 2], [7, 4, 5, 6]], fluxes_with_gyrophase_rotated_by_90[0])
        np.testing.assert_array_almost_equal([[11, 8, 9, 10], [15, 12, 13, 14]], fluxes_with_gyrophase_rotated_by_90[1])

        fluxes_rebinned_with_different_areas = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles, gyrophases,
                                                                                  [[1, 1, 1, 1], [3, 3, 3, 3]], 1, 4)
        np.testing.assert_array_almost_equal(fluxes_rebinned_with_different_areas, [[[3, 4, 5, 6]], [[11, 12, 13, 14]]])

        rebinned_includes_empty_bins = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                          gyrophases,
                                                                          sector_areas,
                                                                          2, 6)
        np.testing.assert_array_almost_equal(rebinned_includes_empty_bins[0],
                                             [[0, np.nan, 1, 2, np.nan, 3], [4, np.nan, 5, 6, np.nan, 7]])

        np.testing.assert_array_almost_equal(rebinned_includes_empty_bins[1],
                                             [[8, np.nan, 9, 10, np.nan, 11], [12, np.nan, 13, 14, np.nan, 15]])

        pitch_angles = np.array([[45, 45, 135, 135], [45, 45, 135, 135]])
        gyrophases = np.array([[135, 225, 225, 135], [45, 315, 315, 45]])

        rebinned_with_varying_pitch_angles = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                                gyrophases,
                                                                                sector_areas,
                                                                                2, 4)

        np.testing.assert_array_almost_equal(rebinned_with_varying_pitch_angles[0],
                                             [[4, 0, 1, 5], [7, 3, 2, 6]])
