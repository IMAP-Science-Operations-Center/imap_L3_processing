import math
from unittest import TestCase

import numpy as np

from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, \
    rebin_by_pitch_angle_and_gyrophase


class TestSectoredProductsAlgorithms(TestCase):

    def test_get_sector_unit_vectors(self):
        sector_vectors = get_sector_unit_vectors([0, 90], [0, 90])
        self.assertEqual((2, 2, 3), sector_vectors.shape)
        np.testing.assert_array_almost_equal(sector_vectors[0], [[0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_almost_equal(sector_vectors[1], [[1, 0, 0], [0, 1, 0]])

    def test_get_hit_bin_polar_coordinates(self):
        declinations, azimuths, declination_delta, azimuth_delta = get_hit_bin_polar_coordinates()

        np.testing.assert_array_equal(declination_delta, [11.25] * 8, strict=True)
        np.testing.assert_array_equal(azimuth_delta, [12.0] * 15, strict=True)
        np.testing.assert_array_almost_equal(declinations, 11.25 + np.arange(0, 8) * 22.5)
        np.testing.assert_array_almost_equal(azimuths, 12 + np.arange(15) * 24)

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
        ]

        for case, pitch_angles, gyrophases, number_of_pitch_angle_bins, number_of_gyrophase_bins, \
                expected_flux_by_pa_gyrophase, expected_flux_by_pa in test_cases:
            with (((self.subTest(case)))):
                rebinned_data = rebin_by_pitch_angle_and_gyrophase(
                    fluxes,
                    flux_delta_plus,
                    flux_delta_minus,
                    pitch_angles,
                    gyrophases,
                    number_of_pitch_angle_bins, number_of_gyrophase_bins)
                fluxes_pa_and_gyro, _, _ = rebinned_data[0:3]
                fluxes_pa_only, _, _ = rebinned_data[3:6]

                np.testing.assert_equal(fluxes_pa_and_gyro, expected_flux_by_pa_gyrophase)
                np.testing.assert_equal(fluxes_pa_only, expected_flux_by_pa)

    def test_rebin_by_pitch_angle_and_gyrophase_includes_uncertainties(self):
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

        rebinned_data = rebin_by_pitch_angle_and_gyrophase(
            fluxes,
            flux_delta_plus,
            flux_delta_minus,
            default_pitch_angles,
            default_gyrophases,
            number_of_pitch_angle_bins, number_of_gyrophase_bins)

        _, delta_plus_pa_gyro, delta_minus_pa_gyro = rebinned_data[0:3]
        _, delta_plus_pa_only, delta_minus_pa_only = rebinned_data[3:6]

        np.testing.assert_allclose(delta_plus_pa_gyro, expected_delta_plus_pa_gyro)
        np.testing.assert_allclose(delta_minus_pa_gyro, expected_delta_minus_pa_gyro)
        np.testing.assert_allclose(delta_plus_pa_only, expected_delta_plus_pa_only, rtol=1e-5)
        np.testing.assert_allclose(delta_minus_pa_only, expected_delta_minus_pa_only, rtol=1e-5)
