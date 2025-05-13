import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup


class TestEnergyLookup(unittest.TestCase):
    def test_read_from_cdf_to_energy_lookup(self):
        expected_bin_centers = np.geomspace(14100, 1)
        actual_energy_lookup = EnergyLookup.from_bin_centers(expected_bin_centers)
        np.testing.assert_array_equal(actual_energy_lookup.bin_centers, expected_bin_centers, )
        
        for i, energy in enumerate(expected_bin_centers):
            self.assertEqual(i, actual_energy_lookup.get_energy_index(energy))
