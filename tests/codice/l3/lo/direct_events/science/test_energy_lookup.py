import unittest

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from tests.test_helpers import get_test_instrument_team_data_path


class TestEnergyLookup(unittest.TestCase):
    def test_energy_lookup_with_imperfect_logspace_values(self):
        l1a_test_path = get_test_instrument_team_data_path(
            "codice/lo/imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf")

        energies = CDF(str(l1a_test_path))["energy_table"][...]

        actual_energy_lookup = EnergyLookup.from_bin_centers(energies)

        np.testing.assert_array_equal(actual_energy_lookup.bin_centers, energies)

        for i, energy in enumerate(energies):
            self.assertEqual(i, actual_energy_lookup.get_energy_index(energy))

        lower_edges = energies - actual_energy_lookup.delta_minus
        upper_edges = energies + actual_energy_lookup.delta_plus

        np.testing.assert_array_equal(lower_edges[1:], upper_edges[:-1])

        self.assertAlmostEqual(np.average(np.diff(np.log(energies))), np.average(np.diff(np.log(lower_edges))),
                               places=5)
        self.assertAlmostEqual(np.average(np.diff(np.log(lower_edges))), np.average(np.diff(np.log(upper_edges))),
                               places=5)
