import unittest
from unittest.mock import sentinel

import numpy as np

from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events


class TestCodiceLoCalculations(unittest.TestCase):
    def test_calculate_partial_densities(self):
        result = calculate_partial_densities(sentinel.intensity)

    def test_calculate_total_number_of_events(self):
        priority_0_tcrs = np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]
                                    ])
        acquisition_time = np.array([[2, 2, 2],
                                     [3, 3, 3],
                                     [1, 1, 1]
                                     ]) * 1_000_000

        expected_total_number_of_events = 12 + 45 + 24

        actual_total_number_of_events = calculate_total_number_of_events(priority_0_tcrs, acquisition_time)
        self.assertEqual(expected_total_number_of_events, actual_total_number_of_events)
