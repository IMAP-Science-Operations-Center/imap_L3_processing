import time
from unittest import TestCase

import numpy as np

from imap_l3_processing.swapi.l3a.science.density_of_neutral_helium_lookup_table import \
    DensityOfNeutralHeliumLookupTable
from tests.test_helpers import get_test_data_path


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

        np.testing.assert_equal(0.0225, density, strict=True)

    def test_get_minimum_distance(self):
        lookup_table = DensityOfNeutralHeliumLookupTable(
            np.array([[0.0, 0.1, .01], [0.0, 0.2, 0.02], [360, 0.1, .015], [360, 0.2, 0.025]]))

        self.assertEqual(0.1, lookup_table.get_minimum_distance())

    def test_perf(self):
        density_of_neutral_helium_lut_path = get_test_data_path('swapi/imap_swapi_l2_density-of-neutral-helium-lut-text-not-cdf_20241023_v002.cdf')

        density_of_neutral_helium_lookup_table = DensityOfNeutralHeliumLookupTable.from_file(
            density_of_neutral_helium_lut_path)

        psi = 4
        distance = np.geomspace(0.0001, 2, 65000)

        t0 = time.perf_counter()
        for i in range(1000):
            density_of_neutral_helium_lookup_table.density(angle=psi, distance=distance)

        print(time.perf_counter() - t0)