import unittest
from unittest.mock import sentinel

import numpy as np

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_coefficient_lookup import MassCoefficientLookup
from imap_l3_processing.codice.l3.lo.models import EnergyAndSpinAngle, PriorityEvent
from imap_l3_processing.codice.l3.lo.science.codice_lo_calculations import calculate_partial_densities, \
    calculate_total_number_of_events, calculate_normalization_ratio, calculate_mass


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

    def test_calculate_mass(self):
        priority_event = PriorityEvent(
            apd_energy=np.array([
                [np.exp(1)], [np.exp(2)],
            ]),
            tof=np.array([
                [np.exp(50)], [np.exp(60)],
            ]),
            apd_gain=np.array([]),
            apd_id=np.array([]),
            data_quality=np.array([]),
            energy_step=np.array([]),
            multi_flag=np.array([]),
            num_events=np.array([]),
            pha_type=np.array([]),
            spin_angle=np.array([]),
            priority_index=1
        )
        lookup = MassCoefficientLookup(np.array([100_000, 10_000, 1000, 100, 10, 1]))
        actual_mass = calculate_mass(priority_event, lookup)

        expected_mass_1 = lookup[0] + lookup[1] * 1 + lookup[2] * 50 + lookup[3] * 1 * 50 + lookup[4] * 1 + lookup[
            5] * np.power(50, 3)
        expected_mass_2 = lookup[0] + lookup[1] * 2 + lookup[2] * 60 + lookup[3] * 2 * 60 + lookup[4] * np.power(2, 2) + \
                          lookup[
                              5] * np.power(60, 3)

        np.testing.assert_array_equal(actual_mass, np.array([[expected_mass_1], [expected_mass_2]]))
