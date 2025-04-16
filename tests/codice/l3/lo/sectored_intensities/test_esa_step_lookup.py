import unittest

import numpy as np

from imap_l3_processing.codice.l3.lo.sectored_intensities.science.esa_step_lookup import ESAStepLookup
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from tests.test_helpers import get_test_data_path


class TestMassPerChargeLookup(unittest.TestCase):
    def test_read_from_file(self):
        esa_step_csv_path = get_test_data_path("codice/esa_step_lookup.csv")

        esa_step_lookup = ESAStepLookup.read_from_file(esa_step_csv_path)

        expected_indices = np.arange(128)
        expected_lookup_values = np.linspace(0.5, 80, 128)
        expected_calibration_factors = np.full(128, 5.75)
        expected_lookup_table = np.array(
            list(zip(expected_indices, expected_lookup_values, expected_calibration_factors)))

        np.testing.assert_array_almost_equal(expected_lookup_table, esa_step_lookup.esa_steps)


if __name__ == '__main__':
    unittest.main()
