import unittest
from unittest.mock import sentinel

import numpy as np

from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events, calculate_normalization_ratio


class TestCodiceLoCalculations(unittest.TestCase):
    def test_calculate_partial_densities(self):
        result = calculate_partial_densities(sentinel.intensity)

    def test_calculate_total_number_of_events(self):
        priority_0_tcrs = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]])
        acquisition_time = np.array([
            [[2, 2, 2],
             [3, 3, 3],
             [1, 1, 1]],
            [[2, 2, 2],
             [3, 3, 3],
             [1, 1, 1]]]) * 1_000_000

        expected_total_number_of_events = [12 + 45 + 24, 12 + 45 + 24]

        actual_total_number_of_events = calculate_total_number_of_events(priority_0_tcrs, acquisition_time)
        np.testing.assert_array_equal(actual_total_number_of_events, expected_total_number_of_events)

    def test_calculate_normalization_ratio(self):
        energy_and_spin_angle_counts = {
            EnergyAndSpinAngle(energy=1, spin_angle=6): 30,
            EnergyAndSpinAngle(energy=2, spin_angle=5): 25,
            EnergyAndSpinAngle(energy=3, spin_angle=4): 20,
        }

        total_number_of_events = 300
        normalization_ratios = calculate_normalization_ratio(energy_and_spin_angle_counts, total_number_of_events)

        expected_normalization_ratio = np.full((128, 12), np.nan)

        expected_normalization_ratio[1][6] = 10
        expected_normalization_ratio[2][5] = 12
        expected_normalization_ratio[3][4] = 15

        np.testing.assert_array_equal(expected_normalization_ratio, normalization_ratios)
