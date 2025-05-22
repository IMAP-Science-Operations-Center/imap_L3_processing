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
            'sw_hplus',
            'sw_heplus2',
            'sw_cplus4',
            'sw_cplus5',
            'sw_cplus6',
            'sw_oplus5',
            'sw_oplus6',
            'sw_oplus7',
            'sw_oplus8',
            'sw_ne',
            'sw_mg',
            'sw_si',
            'sw_felowq',
            'sw_fehighq',
            'sw_heplus',
            'sw_cnoplus',
        ]

        expected_nsw_species = [
            'nsw_hplus',
            'nsw_heplus2',
            'nsw_c',
            'nsw_o',
            'nsw_ne',
            'nsw_simg',
            'nsw_fe',
            'nsw_heplus',
            'nsw_cnoplus',
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

        self.assertEqual(6, lookup.get_num_species())

    def test_get_species_index(self):
        csv_path = get_test_data_path('codice/species_mass_bins.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)

        test_cases = [('sw_hplus', 0, EventDirection.Sunward),
                      ('sw_heplus2', 1, EventDirection.Sunward),
                      ('sw_cplus4', 2, EventDirection.Sunward),
                      ('sw_cplus5', 3, EventDirection.Sunward),
                      ('sw_cplus6', 4, EventDirection.Sunward),
                      ('sw_oplus5', 5, EventDirection.Sunward),
                      ('sw_oplus6', 6, EventDirection.Sunward),
                      ('sw_oplus7', 7, EventDirection.Sunward),
                      ('sw_oplus8', 8, EventDirection.Sunward),
                      ('sw_ne', 9, EventDirection.Sunward),
                      ('sw_mg', 10, EventDirection.Sunward),
                      ('sw_si', 11, EventDirection.Sunward),
                      ('sw_felowq', 12, EventDirection.Sunward),
                      ('sw_fehighq', 13, EventDirection.Sunward),
                      ('sw_heplus', 14, EventDirection.Sunward),
                      ('sw_cnoplus', 15, EventDirection.Sunward),
                      ('nsw_hplus', 16, EventDirection.NonSunward),
                      ('nsw_heplus2', 17, EventDirection.NonSunward),
                      ('nsw_c', 18, EventDirection.NonSunward),
                      ('nsw_o', 19, EventDirection.NonSunward),
                      ('nsw_ne', 20, EventDirection.NonSunward),
                      ('nsw_simg', 21, EventDirection.NonSunward),
                      ('nsw_fe', 22, EventDirection.NonSunward),
                      ('nsw_heplus', 23, EventDirection.NonSunward),
                      ('nsw_cnoplus', 24, EventDirection.NonSunward)]
        for species, index, is_sw in test_cases:
            with self.subTest(species):
                self.assertEqual(index, mass_species_bin_lookup.get_species_index(species, is_sw))
