from dataclasses import replace
from datetime import datetime, timedelta, date
from unittest import TestCase
from unittest.mock import patch, sentinel, call

import numpy as np
from imap_data_access import config
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, \
    FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.descriptors import PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, \
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, \
    EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR, \
    CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR, ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import FittingParameters
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SWAPI_L2_DESCRIPTOR
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import DeltaMinusPlus
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor


class TestSwapiProcessor(TestCase):
    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3PickupIonData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_speed')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_clock_angle')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_deflection_angle')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_ten_minute_velocities')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_pui(self, mock_spicepy, mock_calculate_helium_pui_temperature,
                             mock_calculate_helium_pui_density,
                             mock_calculate_ten_minute_velocities, mock_calculate_pickup_ion,
                             mock_swapi_l3_dependencies_class, mock_calculate_deflection_angle,
                             mock_calculate_clock_angle,
                             mock_calculate_proton_solar_wind_speed, mock_chunk_l2_data, mock_write_cdf,
                             mock_pickup_ion_data_constructor, mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime.now()
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        returned_proton_sw_speed = ufloat(400000, 2)
        mock_calculate_proton_solar_wind_speed.return_value = (
            returned_proton_sw_speed, sentinel.a, sentinel.phi, sentinel.b)

        returned_proton_sw_clock_angle = ufloat(200, 0.25)
        mock_calculate_clock_angle.return_value = returned_proton_sw_clock_angle

        returned_proton_sw_deflection_angle = ufloat(5, 0.001)
        mock_calculate_deflection_angle.return_value = returned_proton_sw_deflection_angle

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        epoch_for_fifty_sweeps = np.arange(initial_epoch, 50)
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)
        chunk_of_fifty = SwapiL2Data(epoch_for_fifty_sweeps, energy * 2, coincidence_count_rate * 2,
                                     coincidence_count_rate_uncertainty * 2)

        expected_fitting_params = FittingParameters(1, 2, 3, 4)
        mock_calculate_pickup_ion.return_value = expected_fitting_params
        mock_calculate_helium_pui_density.return_value = 5
        mock_calculate_helium_pui_temperature.return_value = 6
        mock_calculate_ten_minute_velocities.return_value = np.array([
            [17, 18, 19]
        ])

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        pickup_ion_data = mock_pickup_ion_data_constructor.return_value
        expected_pickup_ion_metadata = replace(input_metadata, descriptor="pui-he")
        pickup_ion_data.input_metadata = expected_pickup_ion_metadata

        input_metadata.descriptor = "pui-he"

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "08" /
                             f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.side_effect = [
            [chunk_of_five],
            [chunk_of_fifty],
        ]

        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.data = sentinel.swapi_l2_data

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        actual_science_input = swapi_processor.dependencies.get_science_inputs()[0]
        self.assertEqual(actual_science_input.get_time_range()[0].strftime("%Y%m%d"), dependency_start_date)

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_instrument_response_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.instrument_response_calibration_table
        mock_geometric_factor_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.geometric_factor_calibration_table
        mock_density_of_neutral_helium_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.density_of_neutral_helium_calibration_table

        mock_chunk_l2_data.side_effect = []

        mock_chunk_l2_data.assert_has_calls([call(sentinel.swapi_l2_data, 5),
                                             call(sentinel.swapi_l2_data, 50)])

        instrument_response_lut, geometric_factor_lut, energies, count_rates, pui_epoch, background_rate_cutoff, \
            sw_velocity_vector, density_of_neutral_helium_lut = mock_calculate_pickup_ion.call_args.args

        self.assertEqual(mock_instrument_response_calibration_table, instrument_response_lut)
        self.assertEqual(mock_geometric_factor_calibration_table, geometric_factor_lut)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        np.testing.assert_array_equal(chunk_of_fifty.energy, energies)
        np.testing.assert_array_equal(chunk_of_fifty.coincidence_count_rate, count_rates)
        self.assertEqual(chunk_of_fifty.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS, pui_epoch)
        self.assertEqual(0.1, background_rate_cutoff)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)

        actual_he_epoch, sw_velocity_vector, density_of_neutral_helium_lut, passed_in_fitting_params = mock_calculate_helium_pui_density.call_args.args

        self.assertEqual(chunk_of_fifty.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS, actual_he_epoch)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        self.assertEqual(expected_fitting_params, passed_in_fitting_params)

        actual_he_epoch, sw_velocity_vector, density_of_neutral_helium_lut, passed_in_fitting_params = mock_calculate_helium_pui_temperature.call_args.args

        self.assertEqual(chunk_of_fifty.epoch[0] + FIVE_MINUTES_IN_NANOSECONDS, actual_he_epoch)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        self.assertEqual(expected_fitting_params, passed_in_fitting_params)

        mock_calculate_ten_minute_velocities.assert_called_with([returned_proton_sw_speed.nominal_value],
                                                                [
                                                                    returned_proton_sw_deflection_angle.nominal_value],
                                                                [returned_proton_sw_clock_angle.nominal_value])
        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_pui-he"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}"),
                                                            ])

        actual_pui_metadata, actual_pui_epoch, actual_pui_cooling_index, actual_pui_ionization_rate, \
            actual_pui_cutoff_speed, actual_pui_background_rate, actual_pui_density, actual_pui_temperature = mock_pickup_ion_data_constructor.call_args.args
        self.assertEqual(expected_pickup_ion_metadata, actual_pui_metadata)
        np.testing.assert_array_equal(np.array([initial_epoch + FIVE_MINUTES_IN_NANOSECONDS]), actual_pui_epoch)
        np.testing.assert_array_equal(np.array([1]), actual_pui_cooling_index)
        np.testing.assert_array_equal(np.array([2]), actual_pui_ionization_rate)
        np.testing.assert_array_equal(np.array([3]), actual_pui_cutoff_speed)
        np.testing.assert_array_equal(np.array([4]), actual_pui_background_rate)
        np.testing.assert_array_equal(np.array([5]), actual_pui_density)
        np.testing.assert_array_equal(np.array([6]), actual_pui_temperature)

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", "pui-he")

        self.assertEqual(input_file_names, pickup_ion_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), pickup_ion_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_speed')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_temperature_and_density')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_clock_angle')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_deflection_angle')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_proton(self, mock_spicepy,
                                mock_swapi_l3_dependencies_class, mock_calculate_deflection_angle,
                                mock_calculate_clock_angle,
                                mock_proton_calculate_temperature_and_density,
                                mock_calculate_proton_solar_wind_speed, mock_chunk_l2_data, mock_write_cdf,
                                mock_proton_solar_wind_data_constructor,
                                mock_imap_attribute_manager
                                ):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime.now()
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        returned_proton_sw_speed = ufloat(400000, 2)
        mock_calculate_proton_solar_wind_speed.return_value = (
            returned_proton_sw_speed, sentinel.a, sentinel.phi, sentinel.b)

        returned_proton_sw_temp = ufloat(99000, 1000)
        returned_proton_sw_density = ufloat(4.97, 0.25)
        mock_proton_calculate_temperature_and_density.return_value = (
            returned_proton_sw_temp, returned_proton_sw_density)

        returned_proton_sw_clock_angle = ufloat(200, 0.25)
        mock_calculate_clock_angle.return_value = returned_proton_sw_clock_angle

        returned_proton_sw_deflection_angle = ufloat(5, 0.001)
        mock_calculate_deflection_angle.return_value = returned_proton_sw_deflection_angle

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        expected_proton_metadata = replace(input_metadata, descriptor="proton-sw")
        proton_solar_wind_data.input_metadata = expected_proton_metadata

        input_metadata.descriptor = "proton-sw"

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "08" /
                             f"imap_swapi_l3a_proton-sw_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.side_effect = [
            [chunk_of_five],
        ]

        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.data = sentinel.swapi_l2_data

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        actual_science_input = swapi_processor.dependencies.get_science_inputs()[0]
        self.assertEqual(actual_science_input.get_time_range()[0].strftime("%Y%m%d"), dependency_start_date)

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_proton_temperature_density_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.proton_temperature_density_calibration_table
        mock_clock_angle_and_flow_deflection_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.clock_angle_and_flow_deflection_calibration_table

        mock_chunk_l2_data.side_effect = []

        mock_chunk_l2_data.assert_has_calls([call(sentinel.swapi_l2_data, 5)])

        expected_count_rate_with_uncertainties = uarray(coincidence_count_rate,
                                                        coincidence_count_rate_uncertainty)
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(
                                          mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(energy, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[1])
        np.testing.assert_array_equal(epoch, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[2])

        self.assertEqual(
            call(mock_clock_angle_and_flow_deflection_calibration_table, returned_proton_sw_speed, sentinel.a,
                 sentinel.phi, sentinel.b), mock_calculate_clock_angle.call_args)

        self.assertEqual(
            call(mock_clock_angle_and_flow_deflection_calibration_table, returned_proton_sw_speed, sentinel.a,
                 sentinel.phi, sentinel.b), mock_calculate_deflection_angle.call_args)

        self.assertEqual(mock_proton_temperature_density_calibration_table,
                         mock_proton_calculate_temperature_and_density.call_args_list[0].args[0])
        self.assert_ufloat_equal(returned_proton_sw_speed,
                                 mock_proton_calculate_temperature_and_density.call_args_list[0].args[1])
        self.assert_ufloat_equal(returned_proton_sw_deflection_angle,
                                 mock_proton_calculate_temperature_and_density.call_args_list[0].args[2])
        self.assertEqual(returned_proton_sw_clock_angle,
                         mock_proton_calculate_temperature_and_density.call_args_list[0].args[3])
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_proton_calculate_temperature_and_density.call_args_list[0].args[
                                              4]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(
                                          mock_proton_calculate_temperature_and_density.call_args_list[0].args[
                                              4]))
        np.testing.assert_array_equal(energy,
                                      mock_proton_calculate_temperature_and_density.call_args_list[0].args[5])

        (actual_proton_metadata, actual_proton_epoch, actual_proton_sw_speed, actual_proton_sw_temperature,
         actual_proton_sw_density, actual_proton_sw_clock_angle,
         actual_proton_sw_deflection_angle) = mock_proton_solar_wind_data_constructor.call_args.args

        self.assertEqual(expected_proton_metadata, actual_proton_metadata)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
                                      actual_proton_epoch,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_speed]), actual_proton_sw_speed, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_temp]), actual_proton_sw_temperature,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_density]), actual_proton_sw_density,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_clock_angle]), actual_proton_sw_clock_angle,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_deflection_angle]),
                                      actual_proton_sw_deflection_angle,
                                      strict=True)

        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_proton-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_proton-sw_{start_date_as_str}_{input_version}"),
                                                            ])

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", "proton-sw")

        self.assertEqual(input_file_names, proton_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), proton_solar_wind_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_alpha_solar_wind_speed')
    @patch(
        'imap_l3_processing.swapi.swapi_processor.calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_alpha(self, mock_spicepy,
                               mock_swapi_l3_dependencies_class, mock_alpha_calculate_temperature_and_density,
                               mock_calculate_alpha_solar_wind_speed,
                               mock_chunk_l2_data, mock_write_cdf,
                               mock_alpha_solar_wind_data_constructor,
                               mock_imap_attribute_manager
                               ):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime.now()
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        returned_alpha_temperature = ufloat(400000, 2000)
        returned_alpha_density = ufloat(0.15, 0.01)
        mock_alpha_calculate_temperature_and_density.return_value = (returned_alpha_temperature, returned_alpha_density)

        returned_alpha_speed = ufloat(450000, 1000)
        mock_calculate_alpha_solar_wind_speed.return_value = ufloat(450000, 1000)

        initial_epoch = 10

        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PROTON_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CLOCK_ANGLE_AND_FLOW_DEFLECTION_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        expected_alpha_metadata = replace(input_metadata, descriptor='alpha-sw')
        alpha_solar_wind_data.input_metadata = expected_alpha_metadata

        descriptor_to_generate, expected_data_product = "alpha-sw", alpha_solar_wind_data

        input_metadata.descriptor = descriptor_to_generate

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "08" /
                             f"imap_swapi_l3a_{descriptor_to_generate}_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.side_effect = [
            [chunk_of_five],
        ]

        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.data = sentinel.swapi_l2_data

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        actual_science_input = swapi_processor.dependencies.get_science_inputs()[0]
        self.assertEqual(actual_science_input.get_time_range()[0].strftime("%Y%m%d"), dependency_start_date)

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_alpha_temperature_density_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.alpha_temperature_density_calibration_table

        mock_chunk_l2_data.assert_has_calls([call(sentinel.swapi_l2_data, 5)])

        expected_count_rate_with_uncertainties = uarray(coincidence_count_rate,
                                                        coincidence_count_rate_uncertainty)

        self.assertEqual(mock_alpha_temperature_density_calibration_table,
                         mock_alpha_calculate_temperature_and_density.call_args_list[0].args[0])
        self.assert_ufloat_equal(returned_alpha_speed,
                                 mock_alpha_calculate_temperature_and_density.call_args_list[0].args[1])
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_alpha_calculate_temperature_and_density.call_args_list[0].args[
                                              2]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(
                                          mock_alpha_calculate_temperature_and_density.call_args_list[0].args[
                                              2]))
        np.testing.assert_array_equal(energy,
                                      mock_alpha_calculate_temperature_and_density.call_args_list[0].args[3])

        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[1])

        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_{descriptor_to_generate}"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_{descriptor_to_generate}_{start_date_as_str}_{input_version}"),
                                                            ])

        actual_alpha_metadata, actual_alpha_epoch, actual_alpha_sw_speed, actual_alpha_sw_temperature, actual_alpha_sw_density = mock_alpha_solar_wind_data_constructor.call_args.args
        self.assertEqual(expected_alpha_metadata, actual_alpha_metadata)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
                                      actual_alpha_epoch)
        np.testing.assert_array_equal(np.array([mock_calculate_alpha_solar_wind_speed.return_value]),
                                      actual_alpha_sw_speed)
        np.testing.assert_array_equal(np.array([mock_alpha_calculate_temperature_and_density.return_value[0]]),
                                      actual_alpha_sw_temperature)
        np.testing.assert_array_equal(np.array([mock_alpha_calculate_temperature_and_density.return_value[1]]),
                                      actual_alpha_sw_density)

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", descriptor_to_generate)

        self.assertEqual(input_file_names, expected_data_product.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), expected_data_product, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.swapi.swapi_processor.calculate_delta_minus_plus')
    @patch('imap_l3_processing.swapi.swapi_processor.save_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3BCombinedVDF')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_combined_sweeps')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3BDependencies')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_alpha_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_proton_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_pui_solar_wind_vdf')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_combined_solar_wind_differential_flux')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3b(self, mock_spiceypy, mock_calculate_combined_solar_wind_differential_flux,
                         mock_calculate_pui_solar_wind_vdf,
                         mock_calculate_proton_solar_wind_vdf,
                         mock_calculate_alpha_solar_wind_vdf,
                         mock_swapi_l3b_dependencies_class,
                         mock_chunk_l2_data,
                         mock_calculate_combined_sweeps, mock_combined_vdf_data,
                         mock_save_data,
                         mock_calculate_delta_minus_plus):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        end_date = datetime.now()
        version = 'v001'
        outgoing_data_level = "l3b"
        dependency_start_date = "20250101"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

        mock_spiceypy.ktotal.return_value = 0
        mock_calculate_proton_solar_wind_vdf.side_effect = [
            (sentinel.proton_calculated_velocities1, sentinel.proton_calculated_probabilities1),
            (sentinel.proton_calculated_velocities2, sentinel.proton_calculated_probabilities2),
        ]

        mock_calculate_alpha_solar_wind_vdf.side_effect = [
            (sentinel.alpha_calculated_velocities1, sentinel.alpha_calculated_probabilities1),
            (sentinel.alpha_calculated_velocities2, sentinel.alpha_calculated_probabilities2),
        ]

        mock_calculate_pui_solar_wind_vdf.side_effect = [
            (sentinel.pui_calculated_velocities1, sentinel.pui_calculated_probabilities1),
            (sentinel.pui_calculated_velocities2, sentinel.pui_calculated_probabilities2),
        ]

        mock_calculate_combined_solar_wind_differential_flux.side_effect = [
            sentinel.calculated_diffential_flux1,
            sentinel.calculated_diffential_flux2
        ]

        mock_calculate_delta_minus_plus.side_effect = [
            DeltaMinusPlus(sentinel.proton_velocity_delta_minus1, sentinel.proton_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.alpha_velocity_delta_minus1, sentinel.alpha_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.pui_velocity_delta_minus1, sentinel.pui_velocity_delta_plus1),
            DeltaMinusPlus(sentinel.energy_delta_minus1, sentinel.energy_delta_plus1),
            DeltaMinusPlus(sentinel.proton_velocity_delta_minus2, sentinel.proton_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.alpha_velocity_delta_minus2, sentinel.alpha_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.pui_velocity_delta_minus2, sentinel.pui_velocity_delta_plus2),
            DeltaMinusPlus(sentinel.energy_delta_minus2, sentinel.energy_delta_plus2),
        ]

        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        average_coincident_count_rates = [14, 15, 16, 17, 18]

        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])
        average_coincident_count_rate_uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]

        first_chunk_initial_epoch = 10
        first_l2_data_chunk = SwapiL2Data(np.array([first_chunk_initial_epoch, 11, 12, 13]), sentinel.energies,
                                          coincidence_count_rate,
                                          coincidence_count_rate_uncertainty)

        second_chunk_initial_epoch = 60
        second_l2_data_chunk = SwapiL2Data(np.array([second_chunk_initial_epoch, 11, 12, 13]), sentinel.energies,
                                           coincidence_count_rate,
                                           coincidence_count_rate_uncertainty)

        mock_chunk_l2_data.return_value = [first_l2_data_chunk, second_l2_data_chunk]

        mock_calculate_combined_sweeps.return_value = [
            uarray(average_coincident_count_rates, average_coincident_count_rate_uncertainties), energy]

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{incoming_data_level}_{GEOMETRIC_FACTOR_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{incoming_data_level}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        science_inputs = [ScienceInput(file_name) for file_name in input_file_names]

        dependencies = ProcessingInputCollection(*science_inputs)

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3b_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_geometric_factor_calibration_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.geometric_factor_calibration_table
        mock_efficiency_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.efficiency_calibration_table

        mock_chunk_l2_data.assert_called_with(mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.data,
                                              50)

        np.testing.assert_array_equal(coincidence_count_rate,
                                      nominal_values(mock_calculate_combined_sweeps.call_args_list[0].args[0]))
        np.testing.assert_array_equal(coincidence_count_rate_uncertainty,
                                      std_devs(mock_calculate_combined_sweeps.call_args_list[0].args[0]))
        self.assertEqual(sentinel.energies,
                         mock_calculate_combined_sweeps.call_args_list[0].args[1])
        mock_efficiency_table.get_efficiency_for.assert_has_calls(
            [call(first_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS),
             call(second_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS)])

        expected_count_rate_with_uncertainties = uarray(average_coincident_count_rates,
                                                        average_coincident_count_rate_uncertainties)

        np.testing.assert_array_equal(energy, mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[0])
        nominal_count_rates = nominal_values(expected_count_rate_with_uncertainties)
        std_devs_count_rates = std_devs(expected_count_rate_with_uncertainties)

        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.get_efficiency_for.return_value,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))

        self.assertEqual(mock_efficiency_table.get_efficiency_for.return_value,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.get_efficiency_for.return_value,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy,
                                      mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates, nominal_values(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates, std_devs(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.get_efficiency_for.return_value,
                         mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[3])
        mock_calculate_delta_minus_plus.assert_has_calls([
            call(sentinel.proton_calculated_velocities1),
            call(sentinel.alpha_calculated_velocities1),
            call(sentinel.pui_calculated_velocities1),
            call(energy),
            call(sentinel.proton_calculated_velocities2),
            call(sentinel.alpha_calculated_velocities2),
            call(sentinel.pui_calculated_velocities2),
            call(energy),
        ])

        expected_combined_metadata = InputMetadata(descriptor="combined", data_level=outgoing_data_level,
                                                   start_date=start_date, end_date=end_date, instrument=instrument,
                                                   version=outgoing_version)

        self.assertEqual(expected_combined_metadata, mock_combined_vdf_data.call_args_list[0].kwargs["input_metadata"])

        np.testing.assert_array_equal(np.array([first_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS,
                                                second_chunk_initial_epoch + FIVE_MINUTES_IN_NANOSECONDS]),
                                      mock_combined_vdf_data.call_args_list[0].kwargs["epoch"])

        np.testing.assert_array_equal([sentinel.proton_calculated_velocities1, sentinel.proton_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["proton_sw_velocities"])
        np.testing.assert_array_equal([sentinel.proton_velocity_delta_minus1, sentinel.proton_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "proton_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.proton_velocity_delta_plus1, sentinel.proton_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "proton_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.proton_calculated_probabilities1, sentinel.proton_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["proton_sw_combined_vdf"])

        np.testing.assert_array_equal([sentinel.alpha_calculated_velocities1, sentinel.alpha_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["alpha_sw_velocities"])
        np.testing.assert_array_equal([sentinel.alpha_velocity_delta_minus1, sentinel.alpha_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "alpha_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.alpha_velocity_delta_plus1, sentinel.alpha_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "alpha_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.alpha_calculated_probabilities1, sentinel.alpha_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["alpha_sw_combined_vdf"])

        np.testing.assert_array_equal([sentinel.pui_calculated_velocities1, sentinel.pui_calculated_velocities2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["pui_sw_velocities"])
        np.testing.assert_array_equal([sentinel.pui_velocity_delta_minus1, sentinel.pui_velocity_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "pui_sw_velocities_delta_minus"])
        np.testing.assert_array_equal([sentinel.pui_velocity_delta_plus1, sentinel.pui_velocity_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs[
                                          "pui_sw_velocities_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.pui_calculated_probabilities1, sentinel.pui_calculated_probabilities2],
            mock_combined_vdf_data.call_args_list[0].kwargs["pui_sw_combined_vdf"])

        np.testing.assert_array_equal([energy, energy],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy"])
        np.testing.assert_array_equal([sentinel.energy_delta_minus1, sentinel.energy_delta_minus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy_delta_minus"])
        np.testing.assert_array_equal([sentinel.energy_delta_plus1, sentinel.energy_delta_plus2],
                                      mock_combined_vdf_data.call_args_list[0].kwargs["combined_energy_delta_plus"])
        np.testing.assert_array_equal(
            [sentinel.calculated_diffential_flux1, sentinel.calculated_diffential_flux2],
            mock_combined_vdf_data.call_args_list[0].kwargs["combined_differential_flux"])

        self.assertEqual(input_file_names, mock_combined_vdf_data.return_value.parent_file_names)
        mock_save_data.assert_called_once_with(mock_combined_vdf_data.return_value)
        self.assertEqual([mock_save_data.return_value], product)

    def assert_ufloat_equal(self, expected_ufloat, actual_ufloat):
        self.assertEqual(expected_ufloat.n, actual_ufloat.n)
        self.assertEqual(expected_ufloat.s, actual_ufloat.s)
