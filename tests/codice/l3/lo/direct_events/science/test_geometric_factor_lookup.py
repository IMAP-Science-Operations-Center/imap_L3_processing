from unittest import TestCase

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.geometric_factor_lookup import GeometricFactorLookup
from tests.test_helpers import get_test_data_folder


class TestGeometricFactorLookup(TestCase):

    def test_geometric_factor_lookup(self):
        geometric_factor_lookup = GeometricFactorLookup(2, 0.5)
        expected_esa_step_end_index = \
            np.array(
                [0, 1, 2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 27, 31, 35, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85, 91, 97,
                 103, 109, 115, 121, 127])
        np.testing.assert_array_equal(geometric_factor_lookup._esa_step_end_index, expected_esa_step_end_index)

    def test_read_from_csv(self):
        path = get_test_data_folder() / 'codice' / 'imap_codice_lo-geometric-factors_20241110_v001.csv'
        lookup = GeometricFactorLookup.from_csv(path)

        self.assertIsInstance(lookup, GeometricFactorLookup)
        self.assertEqual(lookup._full_factor, 2)
        self.assertEqual(lookup._reduced_factor, 0.5)

    def test_get_geometric_factors(self):
        num_epochs = 3
        num_energies = 128

        rgfo_half_spin = np.array([3, 8, 12])

        full_geometric_factor = 2.0
        reduced_geometric_factor = 1.0
        geometric_factor_lookup = GeometricFactorLookup(
            _full_factor=full_geometric_factor,
            _reduced_factor=reduced_geometric_factor,
        )

        actual_geometric_factors = geometric_factor_lookup.get_geometric_factors(rgfo_half_spin)

        self.assertEqual((num_epochs, num_energies), actual_geometric_factors.shape)

        self.assertTrue(np.all(actual_geometric_factors[0, :3] == full_geometric_factor))
        self.assertTrue(np.all(actual_geometric_factors[0, 3:] == reduced_geometric_factor))

        self.assertTrue(np.all(actual_geometric_factors[1, :12] == full_geometric_factor))
        self.assertTrue(np.all(actual_geometric_factors[1, 12:] == reduced_geometric_factor))

        self.assertTrue(np.all(actual_geometric_factors[2, :24] == full_geometric_factor))
        self.assertTrue(np.all(actual_geometric_factors[2, 24:] == reduced_geometric_factor))
