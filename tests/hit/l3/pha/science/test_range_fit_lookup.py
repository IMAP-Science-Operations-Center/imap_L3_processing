import csv
import os
import tempfile
import unittest

import numpy as np

from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, DetectorSide, DetectorRange
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import double_power_law, RangeFitLookup
from tests.test_helpers import get_test_data_path


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
        test_cases = [
            (DetectedRange(DetectorRange.R2, DetectorSide.A), "range2A.csv"),
            (DetectedRange(DetectorRange.R2, DetectorSide.B), "range2B.csv"),
            (DetectedRange(DetectorRange.R3, DetectorSide.A), "range3A.csv"),
            (DetectedRange(DetectorRange.R3, DetectorSide.B), "range3B.csv"),
            (DetectedRange(DetectorRange.R4, DetectorSide.A), "range4A.csv"),
            (DetectedRange(DetectorRange.R4, DetectorSide.B), "range4B.csv"),
        ]
        for range, file_to_populate in test_cases:
            with self.subTest(range):
                with tempfile.TemporaryDirectory() as tempdir:
                    range2A_path = os.path.join(tempdir, "range2A.csv")
                    range3A_path = os.path.join(tempdir, "range3A.csv")
                    range4A_path = os.path.join(tempdir, "range4A.csv")
                    range2B_path = os.path.join(tempdir, "range2B.csv")
                    range3B_path = os.path.join(tempdir, "range3B.csv")
                    range4B_path = os.path.join(tempdir, "range4B.csv")

                    expected_charges = [12, 13]
                    charge1_parameters = [1, 2, 3, 4, 5]
                    charge2_parameters = [4, 5, 6, 7, 8]

                    open(range2A_path, 'w').close()
                    open(range3A_path, 'w').close()
                    open(range4A_path, 'w').close()
                    open(range2B_path, 'w').close()
                    open(range3B_path, 'w').close()
                    open(range4B_path, 'w').close()

                    path_to_populate = os.path.join(tempdir, file_to_populate)
                    with open(path_to_populate, 'w') as range_file:
                        range_file.write("""# 
# Format: ion_charge, a1, b1, a2, b2, gamma
# 
""")
                        csv_writer = csv.writer(range_file)
                        csv_writer.writerow(expected_charges[0:1] + charge1_parameters)
                        csv_writer.writerow(expected_charges[1:2] + charge2_parameters)

                    lookup_table = RangeFitLookup.from_files(
                        range2A_file=range2A_path,
                        range3A_file=range3A_path,
                        range4A_file=range4A_path,
                        range2B_file=range2B_path,
                        range3B_file=range3B_path,
                        range4B_file=range4B_path
                    )

                    e_prime = 11.2

                    charges, delta_e_loss = lookup_table.evaluate_e_prime(range, e_prime)

                    np.testing.assert_equal(charges, expected_charges)
                    np.testing.assert_equal([double_power_law(e_prime, *charge1_parameters),
                                             double_power_law(e_prime, *charge2_parameters)], delta_e_loss)

    def test_load_from_real_example_files(self):
        range_fit_lookup = RangeFitLookup.from_files(
            get_test_data_path("hit/pha_events/imap_hit_l3_range2A-fit-text-not-cdf_20250203_v001.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_l3_range3A-fit-text-not-cdf_20250203_v001.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_l3_range4A-fit-text-not-cdf_20250203_v001.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_l3_range2B-fit-text-not-cdf_20250203_v001.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_l3_range3B-fit-text-not-cdf_20250203_v001.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_l3_range4B-fit-text-not-cdf_20250203_v001.cdf"),
        )

        self.assertIsInstance(range_fit_lookup, RangeFitLookup)
