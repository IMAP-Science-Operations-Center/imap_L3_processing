import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup, \
    PositionToElevationLookup


class TestCoDICEAngleLookup(unittest.TestCase):
    def test_spin_angle_lookup(self):
        spin_angle_lut = SpinAngleLookup()

        self.assertEqual(24, len(spin_angle_lut.bin_deltas))

        lower_edges = spin_angle_lut.bin_centers - spin_angle_lut.bin_deltas
        upper_edges = spin_angle_lut.bin_centers + spin_angle_lut.bin_deltas

        np.testing.assert_array_equal(spin_angle_lut.lower_bin_edges, lower_edges)
        np.testing.assert_array_equal(lower_edges[1:], upper_edges[:-1])

        self.assertEqual(0, lower_edges[0])
        self.assertEqual(360, upper_edges[-1])

        for i in range(24):
            self.assertEqual(i, spin_angle_lut.get_spin_angle_index(15.0 * i + 7.5))

    def test_elevation_lookup(self):
        elevation_angle_lut = PositionToElevationLookup()

        self.assertEqual(13, len(elevation_angle_lut.bin_deltas))

        lower_edges = elevation_angle_lut.bin_centers - elevation_angle_lut.bin_deltas
        upper_edges = elevation_angle_lut.bin_centers + elevation_angle_lut.bin_deltas

        np.testing.assert_array_equal(elevation_angle_lut.lower_bin_edges, lower_edges)
        np.testing.assert_array_equal(lower_edges[1:], upper_edges[:-1])

        self.assertEqual(-7.5, lower_edges[0])
        self.assertEqual(187.5, upper_edges[-1])

        for i in range(13):
            self.assertEqual(i, elevation_angle_lut.get_elevation_index(15.0 * i))

    def test_apd_to_elevation_index(self):
        elevation_angle_lut = PositionToElevationLookup()
        actual_indices = []
        for apd_id in range(1, 25):
            actual_indices.append(elevation_angle_lut.apd_to_elevation_index(apd_id))
        expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        np.testing.assert_array_equal(actual_indices, expected_indices)

    def test_apd_to_elevation(self):
        elevation_angle_lut = PositionToElevationLookup()
        actual_elevations = []
        for apd_id in range(1, 25):
            actual_elevations.append(elevation_angle_lut.apd_to_elevation(apd_id))
        expected_elevations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 165, 150, 135, 120, 105, 90, 75,
                               60, 45, 30, 15]
        np.testing.assert_array_equal(actual_elevations, expected_elevations)
