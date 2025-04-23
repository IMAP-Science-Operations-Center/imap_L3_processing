import csv
import tempfile
import unittest
from pathlib import Path

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup


class TestMassCoefficientLookup(unittest.TestCase):
    def test_mass_coefficient_lookup(self):
        mass_coefficient_lookup_csv_path = "mass_coefficient_lookup.csv"
        coefficients = [[5.29633], [-1.5053], [-2.86483], [0.473693], [0.0900633], [0.0783456]]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv_file = Path(tmpdir) / mass_coefficient_lookup_csv_path
            with open(temp_csv_file, "w") as csv_file:
                csvwriter = csv.writer(csv_file)
                csvwriter.writerows(coefficients)

            lookup = MassCoefficientLookup.read_from_csv(temp_csv_file)

            for i, coefficient in enumerate(coefficients):
                self.assertEqual(coefficient, lookup[i])
