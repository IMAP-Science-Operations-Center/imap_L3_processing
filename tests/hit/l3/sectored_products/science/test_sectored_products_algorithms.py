import math
from unittest import TestCase

import numpy as np

from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, \
    rebin_by_pitch_angle_and_gyrophase


class TestSectoredProductsAlgorithms(TestCase):

    def test_get_sector_unit_vectors(self):
        sector_vectors = get_sector_unit_vectors([0, 90], [0, 90])
        self.assertEqual((2, 2, 3), sector_vectors.shape)
        np.testing.assert_array_almost_equal([[0, 0, 1], [0, 0, 1]], sector_vectors[0])
        np.testing.assert_array_almost_equal([[1, 0, 0], [0, 1, 0]], sector_vectors[1])

    def test_get_hit_bin_polar_coordinates(self):
        declinations, azimuths, declination_delta, azimuth_delta = get_hit_bin_polar_coordinates()

        self.assertEqual(11.25, declination_delta)
        self.assertEqual(12, azimuth_delta)
        np.testing.assert_array_almost_equal(11.25 + np.arange(0, 8) * 22.5, declinations)
        np.testing.assert_array_almost_equal(12 + np.arange(15) * 24, azimuths)

    def test_rebin_by_pitch_angle_and_gyrophase(self):
        default_pitch_angles = np.array([[45, 45, 45, 45], [135, 135, 135, 135]])
        default_gyrophases = np.array([[45, 135, 225, 315], [45, 135, 225, 315]])
        flux_data_for_energy_1 = [[0, 1, 2, 3], [4, 5, 6, 7]]
        flux_data_for_energy_2 = [[8, 9, 10, 11], [12, 13, 14, 15]]
        fluxes = np.array([
            flux_data_for_energy_1,
            flux_data_for_energy_2
        ])
        flux_delta_plus = fluxes * 0.09
        flux_delta_minus = fluxes * 0.11

        test_cases = [
            ('One bin', default_pitch_angles, default_gyrophases, 1, 1, [[[3.5]], [[11.5]]], [[3.5], [11.5]]),
            ('Rebinned to same bins', default_pitch_angles, default_gyrophases, 2, 4, fluxes,
             np.array([[6, 22], [38, 54]])),
            ('Gyrophase rotated by 90', default_pitch_angles, (default_gyrophases + 90) % 360, 2, 4,
             [
                 [[3, 0, 1, 2], [7, 4, 5, 6]],
                 [[11, 8, 9, 10], [15, 12, 13, 14]]
             ],
             [[6, 22], [38, 54]]),
            ('Includes empty bins', default_pitch_angles, default_gyrophases, 2, 6,
             [
                 [[0, np.nan, 1, 2, np.nan, 3], [4, np.nan, 5, 6, np.nan, 7]],
                 [[8, np.nan, 9, 10, np.nan, 11], [12, np.nan, 13, 14, np.nan, 15]]
             ],
             [[6, 22], [38, 54]]),
            ('Varying Pitch Angles', np.array([[45, 45, 135, 135], [45, 45, 135, 135]]),
             np.array([[135, 225, 225, 135], [45, 315, 315, 45]]), 2, 4,
             [
                 [[4, 0, 1, 5], [7, 3, 2, 6]],
                 [[12, 8, 9, 13], [15, 11, 10, 14]]
             ],
             [[10, 18], [42, 50]]),
        ]

        for case, pitch_angles, gyrophases, number_of_pitch_angle_bins, number_of_gyrophase_bins, \
                expected_flux_by_pa_gyrophase, expected_flux_by_pa in test_cases:
            with self.subTest(case):
                fluxes_rebinned_to_one_bin, _, _, fluxes_rebinned_to_one_bin_pa_only, _, _ = rebin_by_pitch_angle_and_gyrophase(
                    fluxes,
                    flux_delta_plus,
                    flux_delta_minus,
                    pitch_angles,
                    gyrophases,
                    number_of_pitch_angle_bins, number_of_gyrophase_bins)
                np.testing.assert_equal(fluxes_rebinned_to_one_bin, expected_flux_by_pa_gyrophase)
                np.testing.assert_equal(fluxes_rebinned_to_one_bin_pa_only, expected_flux_by_pa)
