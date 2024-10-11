import unittest
from datetime import datetime
from unittest.mock import patch, Mock, call

import numpy as np

from imap_processing.constants import HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND, \
    HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000, PROTON_MASS_KG, \
    PROTON_CHARGE_COULOMBS, ONE_AU_IN_KM
from imap_processing.spice_wrapper import FAKE_ROTATION_MATRIX_FROM_PSP
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, extract_pui_energy_bins, \
    _model_count_rate_denominator, convert_velocity_relative_to_imap, calculate_velocity_vector, FittingParameters, \
    ForwardModel, convert_velocity_to_reference_frame, model_count_rate, model_count_rate_integral
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

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_calculate_velocity_vector(self, mock_spice):
        mock_spice.sphrec.return_value = np.array([1, 2, 3])
        speed = 45.6787
        colatitude = 88.3
        phi = -149.0
        actual_pui_vector = calculate_velocity_vector(speed, colatitude, phi)
        np.testing.assert_array_equal(np.array([1, 2, 3]), actual_pui_vector)
        mock_spice.sphrec.assert_called_with(speed, colatitude, phi)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_convert_velocity_to_reference_frame(self, mock_spice):
        input_vector = np.array([1, 2, 3])
        ephemeris_time = 10000000

        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        result = convert_velocity_to_reference_frame(input_vector, ephemeris_time, "FROM", "TO")

        column_1 = 1 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 3]
        column_2 = 2 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 4]
        column_3 = 3 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 5]
        expected_result = column_1 + column_2 + column_3

        np.testing.assert_array_almost_equal(expected_result, result)
        mock_spice.sxform.assert_called_with("FROM", "TO", ephemeris_time)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_relative_to_imap")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_forward_model(self, mock_spice, mock_convert_velocity):
        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch

        imap_position_rectangular_coordinates = np.array([50, 60, 70, 0, 0, 0])
        mock_spice.spkezr.return_value = imap_position_rectangular_coordinates
        imap_position_latitudinal_coordinates = np.array([10, 11, 12])
        mock_spice.reclat.return_value = imap_position_latitudinal_coordinates
        pui_velocity_instrument_frame = np.array([8, 6, 4])
        mock_spice.sphrec.return_value = pui_velocity_instrument_frame

        pui_velocity_gse_frame = np.array([5, 7, 9])
        mock_convert_velocity.return_value = pui_velocity_gse_frame

        fitting_parameters = FittingParameters(0.1, 0.47, 42, 23)
        epoch = datetime(2024, 10, 10)
        solar_wind_vector_gse_frame = np.array([1, 2, 3])
        solar_wind_speed_inertial_frame = np.array([4, 5, 6])

        energy = 94
        theta = 75
        phi = -135

        forward_model = ForwardModel(fitting_parameters, epoch, solar_wind_vector_gse_frame,
                                     solar_wind_speed_inertial_frame)
        result = forward_model.f(energy, theta, phi)

        expected_term_1 = 0.1 / (4 * np.pi)
        expected_term_2 = (0.47 * ONE_AU_IN_KM ** 2) / (
                imap_position_latitudinal_coordinates[0] * solar_wind_speed_inertial_frame * 42)
        magnitude = 8.774964387392123
        expected_term_3 = (magnitude / 42) ** (0.1 - 3)
        expected_term_4 = 1
        expected_term_5 = 1

        expected = expected_term_1 * expected_term_2 * expected_term_3 * expected_term_4 * expected_term_5
        np.testing.assert_array_equal(result, expected)
        mock_spice.datetime2et.assert_called_with(epoch)
        mock_spice.spkezr.assert_called_with("IMAP", ephemeris_time_for_epoch, "ECLIPJ2000", "NONE", "SUN")
        np.testing.assert_array_equal(imap_position_rectangular_coordinates[0:3], mock_spice.reclat.call_args.args[0])
        speed = 67.32371
        self.assertAlmostEqual(speed, mock_spice.sphrec.call_args.args[0], 5)
        self.assertAlmostEqual(theta, mock_spice.sphrec.call_args.args[1])
        self.assertAlmostEqual(phi, mock_spice.sphrec.call_args.args[2])
        mock_convert_velocity.assert_called_with(pui_velocity_instrument_frame, ephemeris_time_for_epoch, "IMAP_SWAPI",
                                                 "GSE")

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_to_reference_frame")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.model_count_rate_integral")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.ForwardModel")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.InstrumentResponseLookupTableCollection")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_model_count_rate(self, mock_spice, mock_instrument_response_lut_collection, mock_forward_model,
                              mock_model_count_rate_integral, mock_convert_velocity_to_reference_frame):
        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch

        sw_instrument_frame_vector = np.array([55, 66, 77])
        mock_spice.sphrec.return_value = sw_instrument_frame_vector
        mock_spice.spkezr.return_value = np.array([99, 88, 77, 66, 55, 44])

        expected_sw_gse_vector = np.array([200, 300, 400])
        expected_sw_hci_vector = np.array([150, 250, 350])
        mock_convert_velocity_to_reference_frame.side_effect = [expected_sw_gse_vector, expected_sw_hci_vector]

        expected_integral_result = np.array([500, 600, 700])
        mock_model_count_rate_integral.return_value = expected_integral_result

        lookup_table = InstrumentResponseLookupTable(np.array([103.07800, 105.04500]),
                                                     np.array([2.0, 1.0]),
                                                     np.array([-149.0, -149.0]),
                                                     np.array([0.97411, 0.99269]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([0.0160000000, 0.0160000000]),
                                                     )
        mock_instrument_response_lut_collection.get_table_for_energy_bin.return_value = lookup_table

        fitting_params = FittingParameters(0.23, 0.57, 0.91, 1.23)
        energy_bin_index = 1
        energy_bin_center = 10000

        geometric_table = Mock()
        geometric_table.lookup_geometric_factor.return_value = 6.4e-13

        result = model_count_rate(fitting_params, energy_bin_index, energy_bin_center,
                                  mock_instrument_response_lut_collection,
                                  geometric_table)

        expected_geo_factor = 6.4e-13 / 2
        expected_denominator = (0.97411 * np.cos(np.deg2rad(90 - 2.0)) * 1.0 * 1.0) + \
                               (0.99269 * np.cos(np.deg2rad(90 - 1.0)) * 1.0 * 1.0)
        expected_result = expected_geo_factor * expected_integral_result / expected_denominator
        np.testing.assert_array_equal(expected_result, result)

        actual_fitting_params, actual_epoch, actual_sw_gse_vector, actual_sw_hci_vector = mock_forward_model.call_args.args
        self.assertIs(fitting_params, actual_fitting_params)
        self.assertEqual(datetime(2010, 1, 1), actual_epoch)

        np.testing.assert_array_equal(expected_sw_gse_vector, actual_sw_gse_vector)
        np.testing.assert_array_equal(expected_sw_hci_vector, actual_sw_hci_vector)
        mock_convert_velocity_to_reference_frame.assert_has_calls([
            call(sw_instrument_frame_vector, ephemeris_time_for_epoch, "IMAP_RTN", "GSE"),
            call(sw_instrument_frame_vector, ephemeris_time_for_epoch, "IMAP_RTN", "HCI")
        ])

        mock_model_count_rate_integral.assert_called_with(lookup_table, mock_forward_model.return_value)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.ForwardModel")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.calculate_sw_speed")
    def test_model_count_rate_integral(self, mock_calculate_speed, mock_forward_model):
        expected_speed_row_1 = 421
        expected_speed_row_2 = 124
        mock_calculate_speed.side_effect = [expected_speed_row_1, expected_speed_row_2]

        mock_forward_model.return_value.f.side_effect = [np.array([33.33, 34.44, 35.55]),
                                                         np.array([44.44, 45.55, 46.66])]

        lookup_table = InstrumentResponseLookupTable(np.array([103.07800, 105.04500]),
                                                     np.array([2.0, 1.1]),
                                                     np.array([-149.0, -132.0]),
                                                     np.array([0.97411, 0.99269]),
                                                     np.array([3.0, 2.0]),
                                                     np.array([1.0, 1.5]),
                                                     np.array([0.0160000000, 0.0230000000]),
                                                     )
        result = model_count_rate_integral(lookup_table, mock_forward_model.return_value)

        expected_row_1_colatitude = 90 - 2.0
        expected_row_1_forward_model_f = np.array([33.33, 34.44, 35.55])
        expected_row_1 = 0.0160000000 * expected_row_1_forward_model_f * expected_speed_row_1 ** 4 * 0.97411 * np.cos(
            np.deg2rad(expected_row_1_colatitude)) * 1.0 * 3.0

        expected_row_2_colatitude = 90 - 1.1
        expected_row_2_forward_model_f = np.array([44.44, 45.55, 46.66])
        expected_row_2 = 0.0230000000 * expected_row_2_forward_model_f * expected_speed_row_2 ** 4 * 0.99269 * np.cos(
            np.deg2rad(expected_row_2_colatitude)) * 1.5 * 2.0

        np.testing.assert_array_equal(result, expected_row_1 + expected_row_2)
