import unittest
from datetime import datetime

import numpy as np
from uncertainties import ufloat

from imap_processing.spice_wrapper import fake_spice_context
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, extract_pui_energy_bins, \
    _model_count_rate_denominator
from imap_processing.swapi.l3a.science.calculate_proton_solar_wind_speed import calculate_sw_speed_h_plus
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable


class CalculatePickupIonParameters(unittest.TestCase):

    def test_calculate_pickup_ion_energy_cutoff(self):
        epoch = 1
        solar_wind_bulk_velocity = ufloat(450.0, 1.0)

        energy_cutoff = calculate_pui_energy_cutoff(epoch, solar_wind_bulk_velocity)

        self.assertAlmostEqual(0.00256375, energy_cutoff.n)
        self.assertAlmostEqual(1.46328e-5, energy_cutoff.s, 10)

    def test_extract_pui_energy_bins(self):
        energies = np.array([100, 1000, 1500, 2000, 10000])
        observed_count_rates = np.array([1, 100, 100, 0.09, 200])
        background_count_rate = 0.1
        energy_cutoff = 1400

        extracted_energies, extracted_count_rates = extract_pui_energy_bins(energies, observed_count_rates,
                                                                            energy_cutoff, background_count_rate)
        np.testing.assert_array_equal(np.array([1500, 10000]), extracted_energies)
        np.testing.assert_array_equal(np.array([100, 200]), extracted_count_rates)

    def test_model_count_rate_denominator(self):
        lookup_table = InstrumentResponseLookupTable(np.array([103.07800, 105.04500]),
                                                     np.array([2.0, 1.0]),
                                                     np.array([-149.0, -149.0]),
                                                     np.array([0.97411, 0.99269]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([0.0160000000, 0.0160000000]),
                                                     )
        result = _model_count_rate_denominator(lookup_table)

        expected = 0.97411 * np.cos(np.deg2rad(90 - 2)) * 1.0 * 1.0 + \
                   0.99269 * np.cos(np.deg2rad(90 - 1.0)) * 1.0 * 1.0
        self.assertEqual(expected, result)

    def test_forward_model(self):


