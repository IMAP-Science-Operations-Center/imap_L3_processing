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
        flux_data_for_energy_1 = np.arange(0, 2 * 4).reshape((2, 4))
        flux_data_for_energy_2 = np.arange(2 * 4, 2 * 2 * 4).reshape((2, 4))
        fluxes = np.array([
            flux_data_for_energy_1,
            flux_data_for_energy_2
        ])
        flux_delta_minus = fluxes - fluxes * 0.09
        flux_delta_plus = fluxes + fluxes * 0.11

        """
        [[0, 1, 2, 3], [4, 5, 6, 7],
        [8, 9, 10, 11], [12, 13, 14, 15]]
        """

        pitch_angles = np.array([[45, 45, 45, 45], [135, 135, 135, 135]])
        gyrophases = np.array([[45, 135, 225, 315], [45, 135, 225, 315]])

        fluxes_rebinned_to_one_bin, fluxes_rebinned_to_one_bin_pa_only = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles, gyrophases,
                                                                        1, 1)
        np.testing.assert_equal(fluxes_rebinned_to_one_bin, [[[3.5]], [[11.5]]])
        np.testing.assert_equal(fluxes_rebinned_to_one_bin_pa_only, [[3.5], [11.5]])

        fluxes_rebinned_to_same_bins, fluxes_rebinned_to_same_bins_pa_only = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles, gyrophases,
                                                                 2, 4)
        np.testing.assert_array_almost_equal(fluxes_rebinned_to_same_bins, fluxes)
        np.testing.assert_array_almost_equal(fluxes_rebinned_to_same_bins_pa_only, np.array([[6, 22], [38, 54]]))

        fluxes_with_gyrophase_rotated_by_90,  fluxes_with_gyrophase_rotated_by_90_pa_only = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                                 (gyrophases + 90) % 360,
                                                                                 2, 4)
        np.testing.assert_array_almost_equal([[3, 0, 1, 2], [7, 4, 5, 6]], fluxes_with_gyrophase_rotated_by_90[0])
        np.testing.assert_array_almost_equal([[11, 8, 9, 10], [15, 12, 13, 14]], fluxes_with_gyrophase_rotated_by_90[1])

        rebinned_includes_empty_bins, rebinned_includes_empty_bins_pa_only  = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                          gyrophases,
                                                                          2, 6)
        np.testing.assert_array_almost_equal(rebinned_includes_empty_bins[0],
                                             [[0, np.nan, 1, 2, np.nan, 3], [4, np.nan, 5, 6, np.nan, 7]])

        np.testing.assert_array_almost_equal(rebinned_includes_empty_bins[1],
                                             [[8, np.nan, 9, 10, np.nan, 11], [12, np.nan, 13, 14, np.nan, 15]])
        np.testing.assert_array_almost_equal(rebinned_includes_empty_bins_pa_only,
                                             [[6,22], [38,54]])

        pitch_angles = np.array([[45, 45, 135, 135], [45, 45, 135, 135]])
        gyrophases = np.array([[135, 225, 225, 135], [45, 315, 315, 45]])

        rebinned_with_varying_pitch_angles, rebinned_with_varying_pitch_angles_pa_only = rebin_by_pitch_angle_and_gyrophase(fluxes, pitch_angles,
                                                                                gyrophases,
                                                                                2, 4)

        [[0, 1, 2, 3], [4, 5, 6, 7],
         [8, 9, 10, 11], [12, 13, 14, 15]]
        np.testing.assert_array_almost_equal(rebinned_with_varying_pitch_angles[0],
                                             [[4, 0, 1, 5], [7, 3, 2, 6]])
        np.testing.assert_array_almost_equal(rebinned_with_varying_pitch_angles[1],
                                             [[12, 8, 9, 13], [15, 11, 10, 14]])
        np.testing.assert_array_almost_equal(rebinned_with_varying_pitch_angles_pa_only,
                                             [[10, 18], [42, 50]])

