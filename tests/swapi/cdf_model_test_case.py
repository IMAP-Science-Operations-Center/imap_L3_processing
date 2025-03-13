import unittest

import numpy as np

from imap_l3_processing.models import DataProductVariable


class CdfModelTestCase(unittest.TestCase):
    def assert_variable_attributes(self, variable: DataProductVariable,
                                   expected_data, expected_name,
                                   expected_data_type=None,
                                   expected_record_varying=True):
        self.assertEqual(expected_name, variable.name)
        np.testing.assert_array_equal(variable.value, expected_data, f"mismatch for var name: {variable.name}")
        self.assertEqual(expected_data_type, variable.cdf_data_type, f"mismatch for var name: {variable.name}")
        self.assertEqual(expected_record_varying, variable.record_varying, f"mismatch for var name: {variable.name}")
