import unittest

import numpy as np
from uncertainties import ufloat

from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, extract_pui_energy_bins


class CalculatePickupIonParameters(unittest.TestCase):

    def test_calculate_pickup_ion_energy_cutoff(self):
        epoch = 1
        solar_wind_bulk_velocity = ufloat(450.0, 1.0)

        energy_cutoff = calculate_pui_energy_cutoff(epoch, solar_wind_bulk_velocity)

        self.assertAlmostEqual(0.00256375, energy_cutoff.n)
        self.assertAlmostEqual(-1.46328e-5, energy_cutoff.s, 4)

    def test_extract_pui_energy_bins(self):
        energies = np.array([100, 1000, 1500, 2000, 10000])
        observed_count_rates = np.array([1, 100, 100, 0.09, 200])
        background_count_rate = 0.1
        energy_cutoff = 1400

        extracted_energies, extracted_count_rates = extract_pui_energy_bins(energies, observed_count_rates,
                                                                            energy_cutoff, background_count_rate)
        np.testing.assert_array_equal(np.array([1500, 10000]), extracted_energies)
        np.testing.assert_array_equal(np.array([100, 200]), extracted_count_rates)
