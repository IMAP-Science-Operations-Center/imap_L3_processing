from unittest import TestCase

import numpy as np

from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates


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
