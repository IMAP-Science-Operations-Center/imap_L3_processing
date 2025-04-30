import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from tests.test_helpers import get_test_data_path


class TestMassSpeciesBinLookup(unittest.TestCase):
    def test_reads_in_csv(self):
        csv_path = get_test_data_path('codice/species_mass_bins.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)
        expected_mass_range = [
            (0.0, 1.5), (2.5, 5.0), (10.0, 14.0), (10.0, 14.0), (10.0, 14.0), (14.0, 18.0), (14.0, 18.0), (14.0, 18.0),
            (14.0, 18.0), (18.0, 22.0), (22.0, 26.0), (26.0, 30.0), (45.0, 65.0), (45.0, 65.0), (2.5, 5.0), (6.0, 25.0),
            (0.0, 1.5), (2.5, 5.0), (10.0, 14.0), (14.0, 18.0), (18.0, 22.0), (22.0, 30.0), (45.0, 65.0), (2.5, 5.0),
            (6.0, 25.0)
        ]
        expected_mass_per_charge_range = [
            (0.7, 1.2), (1.5, 2.5), (2.7, 3.2), (2.2, 2.7), (1.8, 2.2), (3.0, 3.5), (2.4, 3.0), (2.1, 2.4), (1.8, 2.1),
            (2.2, 5.0), (2.0, 5.0), (2.0, 6.0), (3.2, 4.5), (4.5, 10.0), (3.5, 5.0), (10.2, 20.0), (0.7, 1.2),
            (1.5, 2.5), (1.8, 3.2), (1.8, 3.5), (2.2, 5.0), (2.0, 5.0), (3.2, 10.0), (3.5, 5.0), (10.2, 20.0),
        ]

        expected_species = [
            "H+", "He++", "C+4", "C+5", "C+6", "O+5", "O+6", "O+7", "O+8", "Ne", "Mg", "Si", "Fe lowQ", "Fe highQ",
            "He+ (PUI)", "CNO+ (PUI)", "H+", "He++", "C", "O", "Ne", "Si and Mg", "Fe", "He+", "CNO+",
        ]

        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['species'], expected_species)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['mass_ranges'], expected_mass_range)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['mass_per_charge_ranges'],
                                      expected_mass_per_charge_range)

    def test_get_species(self):
        species_look_up = {
            "species": ["He+", "CNO", "Mg"],
            "mass_ranges": [(1, 5), (5, 10), (7, 11)],
            "mass_per_charge_ranges": [(100, 500), (500, 1000), (1000, 1500)]
        }

        lookup = MassSpeciesBinLookup(species_look_up)

        self.assertEqual("He+", lookup.get_species(4, 200))
        self.assertEqual("CNO", lookup.get_species(8, 600))
        self.assertEqual("Mg", lookup.get_species(8, 1100))
