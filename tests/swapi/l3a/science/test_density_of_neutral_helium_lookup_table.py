from unittest import TestCase

import numpy as np

from imap_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import DensityOfNeutralHeliumLookupTable


class TestDensityOfNeutralHeliumLookupTable(TestCase):
    def test_neutral_helium_lookup_table(self):
        lookup_table = DensityOfNeutralHeliumLookupTable(
            np.array([[0.0, 0.1, .01], [0.0, 0.2, 0.02], [360, 0.1, .015], [360, 0.2, 0.025]]))

        densities = lookup_table.density(np.array([0., 0., 180.0, 540.0]), np.array([0.5, 0.15, .2, .2]))

        np.testing.assert_array_equal([0.0, 0.015, 0.0225, 0.0225], densities)

    def test_neutral_helium_lookup_table_single_distance(self):
        lookup_table = DensityOfNeutralHeliumLookupTable(
            np.array([[0.0, 0.1, .01], [0.0, 0.2, 0.02], [360, 0.1, .015], [360, 0.2, 0.025]]))

        density = lookup_table.density(180.0, .2)

        self.assertEqual(0.0225, density)

    def test_get_minimum_distance(self):
        lookup_table = DensityOfNeutralHeliumLookupTable(
            np.array([[0.0, 0.1, .01], [0.0, 0.2, 0.02], [360, 0.1, .015], [360, 0.2, 0.025]]))

        self.assertEqual(0.1, lookup_table.get_minimum_distance())
