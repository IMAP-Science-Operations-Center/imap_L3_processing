import unittest
from unittest.mock import patch

import numpy as np

from imap_processing.constants import HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND, \
    HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000, PROTON_MASS_KG, \
    PROTON_CHARGE_COULOMBS
from imap_processing.spice_wrapper import FAKE_ROTATION_MATRIX_FROM_PSP
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, extract_pui_energy_bins, \
    _model_count_rate_denominator, convert_velocity_relative_to_imap
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable


class TestCalculatePickupIon(unittest.TestCase):

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_relative_to_imap")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_calculate_pickup_ion_energy_cutoff(self, mock_spice, mock_convert_velocity):
        expected_ephemeris_time = 0.00032
        mock_spice.datetime2et.return_value = expected_ephemeris_time
        mock_spice.spkezr.return_value = np.array([0, 0, 0, 4, 0, 0])
        mock_spice.latrec.return_value = np.array([0, 2, 0])

        expected_sw_velocity_in_eclipj2000_frame = np.array([1, 2, 4])
        mock_convert_velocity.return_value = expected_sw_velocity_in_eclipj2000_frame

        epoch = 1
        solar_wind_velocity_in_imap_frame = np.array([22, 33, 44])

        energy_cutoff = calculate_pui_energy_cutoff(epoch, solar_wind_velocity_in_imap_frame)

        mock_spice.datetime2et.assert_called_with(epoch)
        mock_spice.spkezr.assert_called_with("IMAP", expected_ephemeris_time, "ECLIPJ2000", "NONE", "SUN")
        mock_spice.latrec.assert_called_with(-HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND,
                                             HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000,
                                             HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000)
        mock_convert_velocity.assert_called_with(solar_wind_velocity_in_imap_frame, expected_ephemeris_time, "IMAP",
                                                 "ECLIPJ2000")

        velocity_cutoff_vector = np.array([-3, 0, 4])
        velocity_cutoff_norm = 5
        self.assertAlmostEqual(0.5 * (PROTON_MASS_KG / PROTON_CHARGE_COULOMBS) * (2 * velocity_cutoff_norm) ** 2,
                               energy_cutoff)

    def test_extract_pui_energy_bins(self):
        energies = np.array([100, 1000, 1500, 2000, 10000])
        energy_indices = np.array([50, 40, 30, 20, 10])
        observed_count_rates = np.array([1, 100, 100, 0.09, 200])
        background_count_rate = 0.1
        energy_cutoff = 1400

        extracted_energy_bin_labels, extracted_energy_bins, extracted_count_rates = extract_pui_energy_bins(
            energy_indices, energies, observed_count_rates,
            energy_cutoff, background_count_rate)
        np.testing.assert_array_equal(np.array([30, 10]), extracted_energy_bin_labels)
        np.testing.assert_array_equal(np.array([1500, 10000]), extracted_energy_bins)
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

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_convert_velocity_relative_to_imap(self, mock_spice):
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP
        mock_spice.spkezr.return_value = np.array([0, 0, 0, 98, 77, 66])

        input_velocity = np.array([12, 34, 45])
        ephemeris_time = 2000
        from_frame = "INPUT_FRAME"
        to_frame = "OUTPUT_FRAME"
        output_velocity = convert_velocity_relative_to_imap(input_velocity, ephemeris_time, from_frame, to_frame)
        expected_velocity = np.array([67.05039482, 57.6104663, 110.62250463])
        np.testing.assert_array_almost_equal(expected_velocity, output_velocity)
        mock_spice.sxform.assert_called_with(from_frame, to_frame, ephemeris_time)
        mock_spice.spkezr.assert_called_with("IMAP", ephemeris_time, to_frame, "NONE", "SUN")
