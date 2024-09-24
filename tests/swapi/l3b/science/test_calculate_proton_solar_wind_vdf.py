from unittest import TestCase

import numpy as np

from imap_processing.constants import PROTON_MASS_KG, PROTON_CHARGE_COULOMBS
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed_h_plus
from imap_processing.swapi.l3b.science.calculate_proton_solar_wind_vdf import calculate_proton_solar_wind_vdf


class TestCalculateProtonSolarWindVDF(TestCase):
    def test_calculate_proton_solar_wind_vdf(self):
        energies = np.array([0, 1000, 750, 500])
        count_rates = np.array([0, 10, 20, 30])
        efficiency = 0.0882
        geometric_factor = 1e-12

        expected_velocities = [0.0, 437.6947142244463, 379.05474162054054, 309.496900517614]
        expected_probabilities = [np.nan, 14874.030602544019, 39664.08160678405, 89244.1836152641]

        velocities, probabilities = calculate_proton_solar_wind_vdf(energies, count_rates, efficiency, geometric_factor)

        np.testing.assert_array_equal(velocities, expected_velocities)
        np.testing.assert_array_equal(probabilities, expected_probabilities)
