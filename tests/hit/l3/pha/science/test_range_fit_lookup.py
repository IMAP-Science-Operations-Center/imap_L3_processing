import csv
import os
import tempfile
import unittest

import numpy as np

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange
from imap_processing.hit.l3.pha.science.range_fit_lookup import double_power_law, RangeFitLookup


class TestRangeFitLookup(unittest.TestCase):

    def test_double_power_law(self):
        a1 = 1.182819e+01
        b1 = 1.865991e-02
        a2 = 6.508012e+01
        b2 = 8.420725e-01
        gamma = 1.508828e+00
        e_prime = 5
        result = double_power_law(e_prime, a1, b1, a2, b2, gamma)

        self.assertAlmostEqual(result, ((a1 * e_prime ** b1) ** gamma + (a2 * e_prime ** b2) ** gamma) ** (1 / gamma))

    def test_loads_csv_into_array(self):
        with tempfile.TemporaryDirectory() as tempdir:
            range2_path = os.path.join(tempdir, "range2.csv")
            range3_path = os.path.join(tempdir, "range3.csv")
            range4_path = os.path.join(tempdir, "range4.csv")

            expected_charges = [12, 13]
            charge1_parameters = [1, 2, 3, 4, 5]
            charge2_parameters = [4, 5, 6, 7, 8]

            with open(range3_path, 'w') as range3_file:
                range3_file.write("""# 
# Format: ion_charge, a1, b1, a2, b2, gamma
# 
""")
                csv_writer = csv.writer(range3_file)
                csv_writer.writerow(expected_charges[0:1] + charge1_parameters)
                csv_writer.writerow(expected_charges[1:2] + charge2_parameters)
            open(range2_path, 'w').close()
            open(range4_path, 'w').close()

            lookup_table = RangeFitLookup.from_files(
                range2_file=range2_path,
                range3_file=range3_path,
                range4_file=range4_path)

            e_prime = 11.2

            charges, delta_e_loss = lookup_table.evaluate_e_prime(DetectedRange.R3, e_prime)

            np.testing.assert_equal(charges, expected_charges)
            np.testing.assert_equal([double_power_law(e_prime, *charge1_parameters),
                                     double_power_law(e_prime, *charge2_parameters)], delta_e_loss)
