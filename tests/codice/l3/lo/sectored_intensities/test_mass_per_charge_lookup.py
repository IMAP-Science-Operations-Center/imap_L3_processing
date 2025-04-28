import unittest

from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from tests.test_helpers import get_test_data_path


class TestMassPerChargeLookup(unittest.TestCase):
    def test_read_from_file(self):
        mass_per_charge_csv_path = get_test_data_path("codice/test_mass_per_charge_lookup.csv")

        mass_per_charge_lookup = MassPerChargeLookup.read_from_file(mass_per_charge_csv_path)

        self.assertEqual(1, mass_per_charge_lookup.hplus)
        self.assertEqual(2, mass_per_charge_lookup.heplusplus)
        self.assertEqual(3, mass_per_charge_lookup.cplus4)
        self.assertEqual(2.4, mass_per_charge_lookup.cplus5)
        self.assertEqual(2, mass_per_charge_lookup.cplus6)
        self.assertEqual(3.2, mass_per_charge_lookup.oplus5)
        self.assertEqual(2.7, mass_per_charge_lookup.oplus6)
        self.assertEqual(2.28, mass_per_charge_lookup.oplus7)
        self.assertEqual(2, mass_per_charge_lookup.oplus8)
        self.assertEqual(3.5, mass_per_charge_lookup.mg)
        self.assertEqual(4, mass_per_charge_lookup.si)
        self.assertEqual(3.85, mass_per_charge_lookup.fe_loq)
        self.assertEqual(7.25, mass_per_charge_lookup.fe_hiq)


if __name__ == '__main__':
    unittest.main()
