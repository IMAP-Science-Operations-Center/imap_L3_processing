from unittest import TestCase

import numpy as np

from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates


class TestSectoredProductsAlgorithms(TestCase):

    def test_get_sector_unit_vectors(self):
        test_cases = [

            ([0], [0], [[[0, 0, 1]]]),
            ([0], [90], [[[0, 0, 1]]]),
            ([0], [180], [[[0, 0, 1]]]),
            ([0], [270], [[[0, 0, 1]]]),
            # ([0], [360], [[[0, 0, 1]]]),

            ([90], [0], [[[1, 0, 0]]]),
            ([90], [90], [[[0, 1, 0]]]),
            ([90], [180], [[[-1, 0, 0]]]),
            ([90], [270], [[[0, -1, 0]]]),

            ([180], [0], [[[0, 0, -1]]]),
            ([180], [90], [[[0, 0, -1]]]),

        ]

        for elevation, azimuth, expected in test_cases:
            with self.subTest(elevation=elevation, azimuth=azimuth, expected=expected):
                np.testing.assert_array_almost_equal(get_sector_unit_vectors(elevation, azimuth), expected)

    def test_get_hit_bin_polar_coordinates(self):
        declinations, azimuths, declination_delta, azimuth_delta = get_hit_bin_polar_coordinates()

        np.testing.assert_array_equal(declination_delta, [11.25] * 8, strict=True)
        np.testing.assert_array_equal(azimuth_delta, [12.0] * 15, strict=True)
        np.testing.assert_array_almost_equal(declinations, 11.25 + np.arange(0, 8) * 22.5)
        np.testing.assert_array_almost_equal(azimuths, 12 + np.arange(15) * 24)
