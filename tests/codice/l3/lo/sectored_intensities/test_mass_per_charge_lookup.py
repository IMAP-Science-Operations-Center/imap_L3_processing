import unittest

from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from tests.test_helpers import get_test_data_path


class TestMassPerChargeLookup(unittest.TestCase):
    def test_read_from_file(self):
        mass_per_charge_csv_path = get_test_data_path("codice/test_mass_per_charge_lookup.csv")

        mass_per_charge_lookup = MassPerChargeLookup.read_from_file(mass_per_charge_csv_path)

        expected_lookup_values = [(1, "H+", 1),
                                  (2, "He++", 2),
                                  (3, "C+4", 3),
                                  (4, "C+5", 2.4),
                                  (5, "C+6", 2),
                                  (6, "O+5", 3.2),
                                  (7, "O+6", 2.7),
                                  (8, "O+7", 2.28),
                                  (9, "O+8", 2),
                                  (10, "Mg", 3.5),
                                  (11, "Si", 4),
                                  (13, "Fe (low Q)", 3.85),
                                  (14, "Fe (high Q)", 7.25),
                                  ]

        self.assertEqual(expected_lookup_values, mass_per_charge_lookup.mass_per_charge)


if __name__ == '__main__':
    unittest.main()
