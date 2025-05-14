import unittest

from imap_l3_processing.codice.l3.lo.direct_events.science.efficiency_lookup import EfficiencyLookup


class TestEfficiencyLookup(unittest.TestCase):
    def test_efficiency_lookup(self):
        num_species = 2
        num_azimuths = 3
        num_energies = 4

        efficiency_lookup = EfficiencyLookup.create_with_fake_data(num_species, num_azimuths, num_energies)

        self.assertEqual((num_species, num_azimuths, num_energies), efficiency_lookup.efficiency_data.shape)
