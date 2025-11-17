from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np

from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, transform_to_10_minute_chunks
from imap_l3_processing.hit.l3.utils import read_l2_hit_data
from tests.test_helpers import get_test_data_path


class TestSectoredProductsAlgorithms(TestCase):

    def test_get_sector_unit_vectors(self):
        test_cases = [

            ([0], [[0], [270]], [[[0, 0, 1]], [[0, 0, 1]]]),
            ([0], [[90], [180]], [[[0, 0, 1]], [[0, 0, 1]]]),
            ([0], [[180], [90]], [[[0, 0, 1]], [[0, 0, 1]]]),
            ([0], [[270], [0]], [[[0, 0, 1]], [[0, 0, 1]]]),
            # ([0], [360], [[[0, 0, 1]]]),

            ([90], [[0], [270]], [[[1, 0, 0]], [[0, -1, 0]]]),
            ([90], [[90], [180]], [[[0, 1, 0]], [[-1, 0, 0]]]),
            ([90], [[180], [90]], [[[-1, 0, 0]], [[0, 1, 0]]]),
            ([90], [[270], [0]], [[[0, -1, 0]], [[1, 0, 0]]]),

            ([180], [[0], [90]], [[[0, 0, -1]], [[0, 0, -1]]]),
            ([180], [[90], [0]], [[[0, 0, -1]], [[0, 0, -1]]]),

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

    def test_transform_to_10_minute_chunks(self):
        scuffed_l2_input = get_test_data_path("hit/imap_hit_l2_macropixel-intensity_20100106_v001.cdf")
        hit_l2_data = read_l2_hit_data(scuffed_l2_input)

        transformed_hit_data = transform_to_10_minute_chunks(hit_l2_data)

        self.assertEqual((144,), transformed_hit_data.epoch.shape)
        self.assertEqual((144,), transformed_hit_data.epoch_delta.shape)
        self.assertTrue(np.all(transformed_hit_data.epoch_delta == timedelta(minutes=5)))
        self.assertEqual(datetime(2010, 1, 5, 23, 52, 48, 500000), transformed_hit_data.epoch[0])

        self.assertEqual((144, 3, 15, 8), transformed_hit_data.h.shape)
        self.assertEqual((144, 3, 15, 8), transformed_hit_data.delta_plus_h.shape)
        self.assertEqual((144, 3, 15, 8), transformed_hit_data.delta_minus_h.shape)
        self.assertAlmostEqual(4.087e-4, transformed_hit_data.h[0, 0, 0, 0])
        self.assertTrue(np.all(np.logical_not(np.isnan(transformed_hit_data.h))))

        self.assertEqual((144, 2, 15, 8), transformed_hit_data.he4.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_plus_he4.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_minus_he4.shape)
        self.assertAlmostEqual(6.322e-4, transformed_hit_data.he4[0, 0, 0, 0])
        self.assertTrue(np.all(np.logical_not(np.isnan(transformed_hit_data.he4))))

        self.assertEqual((144, 2, 15, 8), transformed_hit_data.cno.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_plus_cno.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_minus_cno.shape)
        self.assertAlmostEqual(7.813e-4, transformed_hit_data.cno[0, 0, 0, 0])
        self.assertTrue(np.all(np.logical_not(np.isnan(transformed_hit_data.cno))))

        self.assertEqual((144, 2, 15, 8), transformed_hit_data.nemgsi.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_plus_nemgsi.shape)
        self.assertEqual((144, 2, 15, 8), transformed_hit_data.delta_minus_nemgsi.shape)
        self.assertAlmostEqual(8.527e-4, transformed_hit_data.nemgsi[0, 0, 0, 0])
        self.assertTrue(np.all(np.logical_not(np.isnan(transformed_hit_data.nemgsi))))

        self.assertEqual((144, 1, 15, 8), transformed_hit_data.fe.shape)
        self.assertEqual((144, 1, 15, 8), transformed_hit_data.delta_plus_fe.shape)
        self.assertEqual((144, 1, 15, 8), transformed_hit_data.delta_minus_fe.shape)
        self.assertAlmostEqual(2.695e-4, transformed_hit_data.fe[0, 0, 0, 0])
        self.assertTrue(np.all(np.logical_not(np.isnan(transformed_hit_data.fe))))
