from unittest import TestCase

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup import GeometricFactorLookup
from tests.test_helpers import get_test_data_folder


class TestGeometricFactorLookup(TestCase):

    def test_default_esa_step_end_index(self):
        geometric_factor_lookup = GeometricFactorLookup(2, 0.5)
        expected_esa_step_end_index = \
            np.array(
                [0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97,
                 103, 109, 115, 121, 127])
        np.testing.assert_array_equal(geometric_factor_lookup._esa_step_end_index, expected_esa_step_end_index)

    def test_read_from_csv(self):
        expected_num_esa_steps = 128
        expected_num_positions = 24

        path = get_test_data_folder() / 'codice' / 'imap_codice_l2-lo-gfactor_20251212_v003.csv'
        lookup = GeometricFactorLookup.read_from_csv(path)

        self.assertIsInstance(lookup, GeometricFactorLookup)

        self.assertEqual((expected_num_positions, expected_num_esa_steps), lookup._full_factor.shape)
        self.assertEqual((expected_num_positions, expected_num_esa_steps), lookup._reduced_factor.shape)
        self.assertEqual(8.71567e-05, lookup._full_factor[0, 0])
        self.assertEqual(9.7615504e-06, lookup._reduced_factor[0, 0])


    def test_get_geometric_factors(self):
        num_epochs = 6
        num_energies = 3
        num_positions = 2

        rgfo_half_spin = np.array([1, 2, 3, 1, 0, 999])
        rgfo_half_spin = np.ma.masked_array(rgfo_half_spin, mask=np.array([False, False, False, False, False, True]))

        full_geometric_factor = np.array([[2, 6, 10], [4, 8, 12]])
        reduced_geometric_factor = np.array([[1, 3, 5], [2, 4, 6]])

        self.assertEqual((num_positions, num_energies), full_geometric_factor.shape)
        self.assertEqual((num_positions, num_energies), reduced_geometric_factor.shape)


        geometric_factor_lookup = GeometricFactorLookup(
            _full_factor=full_geometric_factor,
            _reduced_factor=reduced_geometric_factor,
            _esa_step_end_index=np.array([0, 1, 2])
        )

        actual_geometric_factors = geometric_factor_lookup.get_geometric_factors(rgfo_half_spin)
        self.assertEqual((num_epochs, num_positions, num_energies), actual_geometric_factors.shape)

        expected_geometric_factors = np.array([
            [[2, 3, 5], [4, 4, 6]],
            [[2, 6, 5], [4, 8, 6]],
            [[2, 6, 10], [4, 8, 12]],
            [[2, 3, 5], [4, 4, 6]],
            [[1, 3, 5], [2, 4, 6]],
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ], dtype=float)

        np.testing.assert_equal(actual_geometric_factors, expected_geometric_factors)

        np.testing.assert_equal(actual_geometric_factors[0, :, 0:1], full_geometric_factor[:, 0:1])
        np.testing.assert_equal(actual_geometric_factors[0, :, 1:], reduced_geometric_factor[:, 1:])
        np.testing.assert_equal(actual_geometric_factors[1, :, 0:2], full_geometric_factor[:, 0:2])
        np.testing.assert_equal(actual_geometric_factors[1, :, 2:], reduced_geometric_factor[:, 2:])
        np.testing.assert_equal(actual_geometric_factors[2, :, :], full_geometric_factor[:, :])

        np.testing.assert_equal(actual_geometric_factors[3, :, 0:1], full_geometric_factor[:, 0:1])
        np.testing.assert_equal(actual_geometric_factors[3, :, 1:], reduced_geometric_factor[:, 1:])
        np.testing.assert_equal(actual_geometric_factors[4, :, :], reduced_geometric_factor[:, :])

        np.testing.assert_equal(actual_geometric_factors[5], np.full((2,3), np.nan))