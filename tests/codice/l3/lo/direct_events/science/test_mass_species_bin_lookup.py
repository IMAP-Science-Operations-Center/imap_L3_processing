import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection
from tests.test_helpers import get_test_data_path


class TestMassSpeciesBinLookup(unittest.TestCase):
    def test_reads_in_csv(self):
        csv_path = get_test_data_path('codice/species_mass_bins.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)
        expected_sw_mass_range = [
            (0.0, 1.5), (2.5, 5.0), (10.0, 14.0), (10.0, 14.0), (10.0, 14.0), (14.0, 18.0), (14.0, 18.0), (14.0, 18.0),
            (14.0, 18.0), (18.0, 22.0), (22.0, 26.0), (26.0, 30.0), (45.0, 65.0), (45.0, 65.0), (2.5, 5.0), (6.0, 25.0),

        ]
        expected_sw_mass_per_charge_range = [
            (0.7, 1.2), (1.5, 2.5), (2.7, 3.2), (2.2, 2.7), (1.8, 2.2), (3.0, 3.5), (2.4, 3.0), (2.1, 2.4), (1.8, 2.1),
            (2.2, 5.0), (2.0, 5.0), (2.0, 6.0), (3.2, 4.5), (4.5, 10.0), (3.5, 5.0), (10.2, 20.0),
        ]

        expected_sw_species = [
            "H+", "He++", "C+4", "C+5", "C+6", "O+5", "O+6", "O+7", "O+8", "Ne", "Mg", "Si", "Fe lowQ", "Fe highQ",
            "He+ (PUI)", "CNO+ (PUI)"
        ]

        expected_nsw_species = [
            "H+", "He++", "C", "O", "Ne", "Si and Mg", "Fe", "He+", "CNO+"
        ]

        expected_nsw_mass_per_charge = [
            (0.7, 1.2), (1.5, 2.5), (1.8, 3.2), (1.8, 3.5), (2.2, 5.0), (2.0, 5.0), (3.2, 10.0), (3.5, 5.0),
            (10.2, 20.0)
        ]

        expected_nsw_mass_ranges = [
            (0.0, 1.5), (2.5, 5.0), (10.0, 14.0), (14.0, 18.0), (18.0, 22.0), (22.0, 30.0), (45.0, 65.0), (2.5, 5.0),
            (6.0, 25.0)
        ]

        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['sw_species'], expected_sw_species)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['sw_mass_ranges'],
                                      expected_sw_mass_range)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['sw_mass_per_charge_ranges'],
                                      expected_sw_mass_per_charge_range)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['nsw_species'], expected_nsw_species)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['nsw_mass_ranges'],
                                      expected_nsw_mass_ranges)
        np.testing.assert_array_equal(mass_species_bin_lookup._range_to_species['nsw_mass_per_charge_ranges'],
                                      expected_nsw_mass_per_charge)

    def test_get_species(self):
        species_look_up = {
            "sw_species": ["He++", "CNO+", "Mg+"],
            "sw_mass_ranges": [(1, 5), (5, 10), (7, 11)],
            "sw_mass_per_charge_ranges": [(100, 500), (500, 1000), (1000, 1500)],
            "nsw_species": ["He+", "CNO", "Mg"],
            "nsw_mass_ranges": [(1, 5), (5, 10), (7, 11)],
            "nsw_mass_per_charge_ranges": [(100, 500), (500, 1000), (1000, 1500)],
        }

        lookup = MassSpeciesBinLookup(species_look_up)

        self.assertEqual("He++", lookup.get_species(4, 200, EventDirection.Sunward))
        self.assertEqual("CNO+", lookup.get_species(8, 600, EventDirection.Sunward))
        self.assertEqual("Mg+", lookup.get_species(8, 1100, EventDirection.Sunward))

        self.assertEqual("He+", lookup.get_species(4, 200, EventDirection.NonSunward))
        self.assertEqual("CNO", lookup.get_species(8, 600, EventDirection.NonSunward))
        self.assertEqual("Mg", lookup.get_species(8, 1100, EventDirection.NonSunward))

    def test_get_species_index(self):
        csv_path = get_test_data_path('codice/species_mass_bins.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)

        test_cases = [('H+', 0, EventDirection.Sunward),
                      ('He++', 1, EventDirection.Sunward),
                      ('C+4', 2, EventDirection.Sunward),
                      ('C+5', 3, EventDirection.Sunward),
                      ('C+6', 4, EventDirection.Sunward),
                      ('O+5', 5, EventDirection.Sunward),
                      ('O+6', 6, EventDirection.Sunward),
                      ('O+7', 7, EventDirection.Sunward),
                      ('O+8', 8, EventDirection.Sunward),
                      ('Ne', 9, EventDirection.Sunward),
                      ('Mg', 10, EventDirection.Sunward),
                      ('Si', 11, EventDirection.Sunward),
                      ('Fe lowQ', 12, EventDirection.Sunward),
                      ('Fe highQ', 13, EventDirection.Sunward),
                      ('He+ (PUI)', 14, EventDirection.Sunward),
                      ('CNO+ (PUI)', 15, EventDirection.Sunward),
                      ('H+', 16, EventDirection.NonSunward),
                      ('He++', 17, EventDirection.NonSunward),
                      ('C', 18, EventDirection.NonSunward),
                      ('O', 19, EventDirection.NonSunward),
                      ('Ne', 20, EventDirection.NonSunward),
                      ('Si and Mg', 21, EventDirection.NonSunward),
                      ('Fe', 22, EventDirection.NonSunward),
                      ('He+', 23, EventDirection.NonSunward),
                      ('CNO+', 24, EventDirection.NonSunward)]
        for species, index, is_sw in test_cases:
            with self.subTest(species):
                self.assertEqual(index, mass_species_bin_lookup.get_species_index(species, is_sw))
