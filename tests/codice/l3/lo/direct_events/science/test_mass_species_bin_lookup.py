import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from tests.test_helpers import get_test_data_path


class TestMassSpeciesBinLookup(unittest.TestCase):
    def test_read_csv(self):
        csv_path = get_test_data_path('codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)

        expected_species = ["hplus", "heplus2", "oplus6", "heplus"]
        expected_mass_per_charge = [(0.7, 1.2), (1.5, 2.5), (2.4, 3.0), (3.5, 5.0)]
        expected_mass_range = [(0.0, 1.5), (2.5, 5.0), (14.0, 18.0), (2.5, 5.0)]

        np.testing.assert_array_equal(mass_species_bin_lookup.species, expected_species)
        np.testing.assert_array_equal(mass_species_bin_lookup.mass_per_charge, expected_mass_per_charge)
        np.testing.assert_array_equal(mass_species_bin_lookup.mass_ranges, expected_mass_range)

    def test_get_species(self):
        lookup = MassSpeciesBinLookup(
            species=["hplus", "heplus2", "oplus6", "heplus"],
            mass_per_charge=[(0.7, 1.2), (1.5, 2.5), (2.4, 3.0), (3.5, 5.0)],
            mass_ranges=[(0.0, 1.5), (2.5, 5.0), (14.0, 18.0), (2.5, 5.0)],
        )

        self.assertEqual("hplus", lookup.get_species(1.0, 1.0))
        self.assertEqual("heplus2", lookup.get_species(4.0, 2.0))
        self.assertEqual("oplus6", lookup.get_species(16.0, 2.6))
        self.assertEqual("heplus", lookup.get_species(4.0, 4))

        self.assertIsNone(lookup.get_species(20.0, 6.0))

        self.assertEqual(4, lookup.get_num_species())

    def test_get_species_index(self):
        csv_path = get_test_data_path('codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)

        test_cases = [
            ("hplus", 0),
            ("heplus2", 1),
            ("oplus6", 2),
            ("heplus", 3)
        ]
        for species, index in test_cases:
            with self.subTest(species):
                self.assertEqual(index, mass_species_bin_lookup.get_species_index(species))
