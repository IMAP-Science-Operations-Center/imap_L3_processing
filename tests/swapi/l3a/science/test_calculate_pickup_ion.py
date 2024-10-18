import unittest
from datetime import datetime
from pathlib import Path
from unittest import skip
from unittest.mock import patch, Mock, call

import numpy as np
from spacepy.pycdf import CDF
from uncertainties.unumpy import uarray

import imap_processing
from imap_processing.constants import HYDROGEN_INFLOW_SPEED_IN_KM_PER_SECOND, \
    HYDROGEN_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000, HYDROGEN_INFLOW_LATITUDE_DEGREES_IN_ECLIPJ2000, PROTON_MASS_KG, \
    PROTON_CHARGE_COULOMBS, ONE_AU_IN_KM, HELIUM_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000
from imap_processing.spice_wrapper import FAKE_ROTATION_MATRIX_FROM_PSP
from imap_processing.swapi.l3a.science.calculate_alpha_solar_wind_speed import calculate_combined_sweeps
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_pui_energy_cutoff, extract_pui_energy_bins, \
    _model_count_rate_denominator, convert_velocity_relative_to_imap, calculate_velocity_vector, FittingParameters, \
    ForwardModel, convert_velocity_to_reference_frame, model_count_rate_integral, \
    calculate_pickup_ion_values, ModelCountRateCalculator, calculate_ten_minute_velocities
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_processing.swapi.l3b.science.instrument_response_lookup_table import InstrumentResponseLookupTable, \
    InstrumentResponseLookupTableCollection


class TestCalculatePickupIon(unittest.TestCase):

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_relative_to_imap")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_calculate_pickup_ion_energy_cutoff(self, mock_spice, mock_convert_velocity):
        expected_ephemeris_time = 0.00032
        mock_spice.datetime2et.return_value = expected_ephemeris_time
        mock_light_time = 1233.002
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 4, 0, 0]), mock_light_time)
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
        self.assertAlmostEqual(0.5 * (PROTON_MASS_KG / PROTON_CHARGE_COULOMBS) * (2 * velocity_cutoff_norm * 1000) ** 2,
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

        expected = 0.97411 * np.sin(np.deg2rad(90 - 2)) * 1.0 * 1.0 + \
                   0.99269 * np.sin(np.deg2rad(90 - 1.0)) * 1.0 * 1.0
        self.assertEqual(expected, result)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_convert_velocity_relative_to_imap(self, mock_spice):
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP
        mock_light_time = 12.459
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 98, 77, 66]), mock_light_time)

        input_velocity = np.array([[12, 34, 45], [67, 89, 45]])
        ephemeris_time = 2000
        from_frame = "INPUT_FRAME"
        to_frame = "OUTPUT_FRAME"
        output_velocity = convert_velocity_relative_to_imap(input_velocity, ephemeris_time, from_frame, to_frame)
        expected_velocity = np.array([[67.05039482, 57.6104663, 110.62250463],
                                      [-9.860859, 46.122475, 108.983876]])
        np.testing.assert_array_almost_equal(output_velocity, expected_velocity)
        mock_spice.sxform.assert_called_with(from_frame, to_frame, ephemeris_time)
        mock_spice.spkezr.assert_called_with("IMAP", ephemeris_time, to_frame, "NONE", "SUN")

    def test_calculate_velocity_vector(self):
        speed = [45.6787, 60.123]
        colatitude_degrees = [88.3, 89.3]
        phi = [-149.0, -100.0]

        actual_pui_vector = calculate_velocity_vector(speed, colatitude_degrees, phi)

        expected_pui_vectors = np.array([[-39.137055, -23.515915, 1.355115],
                                         [-10.43947, -59.205178, 0.734523]])
        np.testing.assert_array_almost_equal(actual_pui_vector, expected_pui_vectors)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_convert_velocity_to_reference_frame(self, mock_spice):
        input_vector = np.array([[1, 2, 3], [1, 2, 3]])
        ephemeris_time = 10000000

        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        result = convert_velocity_to_reference_frame(input_vector, ephemeris_time, "FROM", "TO")

        column_1 = 1 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 3]
        column_2 = 2 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 4]
        column_3 = 3 * FAKE_ROTATION_MATRIX_FROM_PSP[3:6, 5]
        expected_result = column_1 + column_2 + column_3

        np.testing.assert_array_almost_equal([expected_result, expected_result], result)
        mock_spice.sxform.assert_called_with("FROM", "TO", ephemeris_time)

        result = convert_velocity_to_reference_frame(input_vector[0], ephemeris_time, "FROM", "TO")
        np.testing.assert_array_almost_equal(expected_result, result)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.DensityOfNeutralHeliumLookupTable.density")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_relative_to_imap")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.calculate_velocity_vector")
    def test_forward_model(self, mock_calculate_velocity_vector, mock_spice, mock_convert_velocity,
                           mock_helium_density):
        mock_helium_density.return_value = 1

        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch

        imap_position_rectangular_coordinates = np.array([50, 60, 70, 0, 0, 0])
        mock_light_time = 123.0
        mock_spice.spkezr.return_value = (imap_position_rectangular_coordinates, mock_light_time)
        imap_position_latitudinal_coordinates = np.array([10, 11, 12])
        mock_spice.reclat.return_value = imap_position_latitudinal_coordinates
        pui_velocity_instrument_frame = np.array([8, 6, 4])
        mock_calculate_velocity_vector.return_value = pui_velocity_instrument_frame

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
        self.assertAlmostEqual(speed, mock_calculate_velocity_vector.call_args.args[0], 5)
        self.assertAlmostEqual(theta, mock_calculate_velocity_vector.call_args.args[1])
        self.assertAlmostEqual(phi, mock_calculate_velocity_vector.call_args.args[2])
        mock_convert_velocity.assert_called_with(pui_velocity_instrument_frame, ephemeris_time_for_epoch, "IMAP_SWAPI",
                                                 "GSE")
        mock_helium_density.assert_called_with(10 * (magnitude / 42) ** 0.1,
                                               11 - HELIUM_INFLOW_LONGITUDE_DEGREES_IN_ECLIPJ2000)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.convert_velocity_to_reference_frame")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.model_count_rate_integral")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.ForwardModel")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.InstrumentResponseLookupTableCollection")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_model_count_rate(self, mock_spice, mock_instrument_response_lut_collection,
                              mock_forward_model,
                              mock_model_count_rate_integral, mock_convert_velocity_to_reference_frame):
        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch

        sw_instrument_frame_vector = np.array([55, 66, 77])
        mock_light_time = 122.0
        mock_spice.spkezr.return_value = (np.array([99, 88, 77, 66, 55, 44]), mock_light_time)

        expected_sw_gse_vector = np.array([200, 300, 400])
        expected_sw_hci_vector = np.array([150, 250, 350])
        mock_convert_velocity_to_reference_frame.side_effect = [expected_sw_gse_vector, expected_sw_hci_vector]

        expected_integral_results = np.array([500, 600, 700])
        mock_model_count_rate_integral.side_effect = expected_integral_results

        lookup_table = InstrumentResponseLookupTable(np.array([103.07800, 105.04500]),
                                                     np.array([2.0, 1.0]),
                                                     np.array([-149.0, -149.0]),
                                                     np.array([0.97411, 0.99269]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([1.0, 1.0]),
                                                     np.array([0.0160000000, 0.0160000000]),
                                                     )
        mock_instrument_response_lut_collection.get_table_for_energy_bin.return_value = lookup_table

        energy_bin_index = 1
        energy_bin_center = 10000

        geometric_table = Mock()
        geometric_table.lookup_geometric_factor.return_value = 6.4e-13

        model_count_rate_calculator = ModelCountRateCalculator(mock_instrument_response_lut_collection, geometric_table,
                                                               sw_instrument_frame_vector)

        fitting_params = FittingParameters(0.23, 0.57, 0.91, 1.23)
        mock_forward_model.return_value.fitting_params = fitting_params

        result = model_count_rate_calculator.model_count_rate([
            (energy_bin_index, energy_bin_center),
            (2, 8000),
            (3, 6000),
        ], fitting_params, datetime(2010, 1, 1))

        expected_geo_factor = 6.4e-13 / 2
        expected_denominator = (0.97411 * np.sin(np.deg2rad(90 - 2.0)) * 1.0 * 1.0) + \
                               (0.99269 * np.sin(np.deg2rad(90 - 1.0)) * 1.0 * 1.0)
        expected_result = expected_geo_factor * expected_integral_results / expected_denominator + fitting_params.background_count_rate
        np.testing.assert_array_equal(expected_result, result)

        actual_fitting_params, actual_epoch, actual_sw_gse_vector, actual_sw_hci_vector_norm = mock_forward_model.call_args.args
        self.assertIs(fitting_params, actual_fitting_params)
        self.assertEqual(datetime(2010, 1, 1), actual_epoch)
        expected_sw_hci_vector_norm = 455.521679
        np.testing.assert_array_equal(expected_sw_gse_vector, actual_sw_gse_vector)
        self.assertAlmostEqual(expected_sw_hci_vector_norm, actual_sw_hci_vector_norm)
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
        mock_calculate_speed.return_value = np.array([expected_speed_row_1, expected_speed_row_2])

        mock_forward_model.return_value.f.return_value = np.array([33.33, 35.55])

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
        expected_row_1_forward_model_f = 33.33
        expected_row_1 = 0.0160000000 * expected_row_1_forward_model_f * expected_speed_row_1 ** 4 * 0.97411 * np.sin(
            np.deg2rad(expected_row_1_colatitude)) * 1.0 * 3.0

        expected_row_2_colatitude = 90 - 1.1
        expected_row_2_forward_model_f = 35.55
        expected_row_2 = 0.0230000000 * expected_row_2_forward_model_f * expected_speed_row_2 ** 4 * 0.99269 * np.sin(
            np.deg2rad(expected_row_2_colatitude)) * 1.5 * 2.0

        np.testing.assert_array_equal(result, expected_row_1 + expected_row_2)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.calculate_pui_energy_cutoff")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.scipy.optimize.minimize")
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_calculate_pickup_ions_with_minimize_mocked(self, mock_spice, mock_minimize,
                                                        mock_calculate_pui_energy_cutoff):
        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch
        mock_light_time = 122.0
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 4, 0, 0]), mock_light_time)
        mock_spice.latrec.return_value = np.array([0, 2, 0])
        mock_spice.reclat.return_value = np.array([1, 0.2, 0.6])
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        data_file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_50-sweeps_20100101_v002.cdf"
        with CDF(str(data_file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

            count_rates_with_uncertainty = uarray(count_rate, count_rate_delta)
            average_count_rates, energies = calculate_combined_sweeps(count_rate, energy)
            response_lut_path = Path(
                imap_processing.__file__).parent.parent / "swapi" / "test_data" / "truncated_swapi_response_simion_v1"

            instrument_response_collection = InstrumentResponseLookupTableCollection(response_lut_path)

            geometric_factor_lut_path = Path(
                imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v001.cdf"

            geometric_factor_lut = GeometricFactorCalibrationTable.from_file(geometric_factor_lut_path)
            background_count_rate_cutoff = 0.1
            epoch = datetime(2024, 10, 17)
            sw_velocity = Mock()
            mock_minimize.return_value.x = [1, 2, 3, 4]
            mock_calculate_pui_energy_cutoff.return_value = 6000.0

            _ = calculate_pickup_ion_values(
                instrument_response_collection, geometric_factor_lut, energies,
                average_count_rates, epoch, background_count_rate_cutoff, sw_velocity)

            mock_calculate_pui_energy_cutoff.assert_called_with(epoch, sw_velocity)
            extracted_count_rates, indices, model_count_rates_calculator, epoch = mock_minimize.call_args.kwargs['args']

            np.testing.assert_array_equal(model_count_rates_calculator.solar_wind_vector, sw_velocity)

    @skip
    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.spiceypy")
    def test_calculate_pickup_ions_with_minimize(self, mock_spice):
        ephemeris_time_for_epoch = 100000
        mock_spice.datetime2et.return_value = ephemeris_time_for_epoch
        mock_light_time = 122.0
        mock_spice.spkezr.return_value = (np.array([0, 0, 0, 4, 0, 0]), mock_light_time)
        mock_spice.latrec.return_value = np.array([0, 2, 0])
        mock_spice.reclat.return_value = np.array([1, 0.2, 0.6])
        mock_spice.sxform.return_value = FAKE_ROTATION_MATRIX_FROM_PSP

        data_file_path = Path(
            imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_50-sweeps_20100101_v002.cdf"
        with CDF(str(data_file_path)) as cdf:
            energy = cdf["energy"][...]
            count_rate = cdf["swp_coin_rate"][...]
            count_rate_delta = cdf["swp_coin_unc"][...]

            count_rates_with_uncertainty = uarray(count_rate, count_rate_delta)
            average_count_rates, energies = calculate_combined_sweeps(count_rate, energy)
            response_lut_path = Path(
                imap_processing.__file__).parent.parent / "swapi" / "test_data" / "swapi_response_simion_v1"

            instrument_response_collection = InstrumentResponseLookupTableCollection(response_lut_path)

            geometric_factor_lut_path = Path(
                imap_processing.__file__).parent.parent / "swapi" / "test_data" / "imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v001.cdf"

            geometric_factor_lut = GeometricFactorCalibrationTable.from_file(geometric_factor_lut_path)
            background_count_rate_cutoff = 0.1
            epoch = datetime(2024, 10, 17)
            actual_fitting_parameters = calculate_pickup_ion_values(
                instrument_response_collection, geometric_factor_lut, energies,
                average_count_rates, epoch, background_count_rate_cutoff)

            mock_spice.datetime2et.assert_called_with(epoch)
            self.assertEqual(1.0549842106923362, actual_fitting_parameters.cooling_index)
            self.assertEqual(1.230969960827135, actual_fitting_parameters.ionization_rate)
            self.assertEqual(1.001668820723139, actual_fitting_parameters.cutoff_speed)
            self.assertEqual(0.18394392413512836, actual_fitting_parameters.background_count_rate)

    @patch("imap_processing.swapi.l3a.science.calculate_pickup_ion.calculate_velocity_vector")
    def test_calculate_ten_minute_velocities(self, mock_calculate_velocity_vector):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])
        z = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                      190, 200, 210])

        mock_calculate_velocity_vector.return_value = np.transpose([x, y, z])

        mock_speed = Mock()
        mock_deflection_angles = Mock()
        mock_clock_angles = Mock()
        averaged_velocities = calculate_ten_minute_velocities(mock_speed, mock_deflection_angles, mock_clock_angles)

        expected_averaged_velocities = np.array([[5.5, 55, 55], [15.5, 155, 155], [21, 210, 210]])

        mock_calculate_velocity_vector.assert_called_with(mock_speed, mock_deflection_angles, mock_clock_angles)

        np.testing.assert_array_equal(expected_averaged_velocities, averaged_velocities)
