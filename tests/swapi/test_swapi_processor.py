from dataclasses import replace
from datetime import datetime, timedelta, date
from unittest import TestCase
from unittest.mock import patch, sentinel, call, Mock

import numpy as np
from imap_data_access import config
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput, AncillaryInput
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_l3_processing.constants import THIRTY_SECONDS_IN_NANOSECONDS, \
    FIVE_MINUTES_IN_NANOSECONDS
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.descriptors import DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR, \
    INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR, EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR, \
    ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR, \
    GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR, GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR, \
    HYDROGEN_INFLOW_VECTOR_DESCRIPTOR, HELIUM_INFLOW_VECTOR_DESCRIPTOR, \
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR, CENTRAL_EFFECTIVE_AREA_DESCRIPTOR, \
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR
from imap_l3_processing.swapi.l3a.models import SwapiL2Data, SwapiL3PickupIonData
from imap_l3_processing.swapi.l3a.science.calculate_pickup_ion import FittingParameters
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SWAPI_L2_DESCRIPTOR, SwapiL3ADependencies
from imap_l3_processing.swapi.l3b.science.calculate_solar_wind_vdf import DeltaMinusPlus
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor, logger


class TestSwapiProcessor(TestCase):
    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3PickupIonData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_ten_minute_velocities')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_pui(self, mock_spicepy, mock_calculate_helium_pui_temperature,
                             mock_calculate_helium_pui_density,
                             mock_calculate_ten_minute_velocities, mock_calculate_pickup_ion,
                             mock_swapi_l3_dependencies_class,
                             mock_parallel_chunk_runner_class,
                             mock_chunk_l2_data, mock_write_cdf,
                             mock_pickup_ion_data_constructor, mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 9, 26)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 9, 25)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        returned_proton_sw_speed = 400000.0
        returned_proton_sw_clock_angle = 200.0
        returned_proton_sw_deflection_angle = 5.0
        runner_result = dict(
            proton_sw_speed=np.array([returned_proton_sw_speed]),
            proton_sw_clock_angle=np.array([returned_proton_sw_clock_angle]),
            proton_sw_deflection_angle=np.array([returned_proton_sw_deflection_angle]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )
        mock_parallel_chunk_runner_class.return_value.run.return_value = runner_result

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
        mock_calculate_ten_minute_velocities.return_value = (np.array([[17, 18, 19]]), np.array([SwapiL3Flags.NONE]))

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{ALPHA_TEMPERATURE_DENSITY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
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

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "09" /
                             f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.side_effect = [
            [chunk_of_five],
            [chunk_of_fifty],
        ]

        mock_l3a_dependencies = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value
        mock_l3a_dependencies.data = Mock(energy=energy)

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        product = swapi_processor.process()

        actual_science_input = swapi_processor.dependencies.get_science_inputs()[0]
        self.assertEqual(actual_science_input.get_time_range()[0].strftime("%Y%m%d"), dependency_start_date)

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_instrument_response_calibration_table = mock_l3a_dependencies.instrument_response_calibration_table
        mock_geometric_factor_calibration_table = mock_l3a_dependencies.geometric_factor_calibration_table
        mock_efficiency_lut = mock_l3a_dependencies.efficiency_calibration_table
        mock_density_of_neutral_helium_calibration_table = mock_l3a_dependencies.density_of_neutral_helium_calibration_table
        mock_hydrogen_inflow_vector = mock_l3a_dependencies.hydrogen_inflow_vector
        mock_helium_inflow_vector = mock_l3a_dependencies.helium_inflow_vector

        mock_chunk_l2_data.side_effect = []

        mock_chunk_l2_data.assert_has_calls([call(mock_l3a_dependencies.data, 5),
                                             call(mock_l3a_dependencies.data, 50)])

        instrument_response_lut, geometric_factor_lut, energies, count_rates, pui_epoch, \
            sw_velocity_vector, density_of_neutral_helium_lut, efficiency_lut, hydrogen_inflow_vector, helium_inflow_vector = mock_calculate_pickup_ion.call_args.args

        self.assertEqual(mock_instrument_response_calibration_table, instrument_response_lut)
        self.assertEqual(mock_efficiency_lut, efficiency_lut)
        self.assertEqual(mock_geometric_factor_calibration_table, geometric_factor_lut)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        np.testing.assert_array_equal(chunk_of_fifty.energy, energies)
        np.testing.assert_array_equal(chunk_of_fifty.coincidence_count_rate, count_rates)
        self.assertEqual(chunk_of_fifty.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS, pui_epoch)
        self.assertEqual(mock_hydrogen_inflow_vector, hydrogen_inflow_vector)
        self.assertEqual(mock_helium_inflow_vector, helium_inflow_vector)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)

        actual_he_epoch, sw_velocity_vector, density_of_neutral_helium_lut, passed_in_fitting_params, helium_inflow_vector = mock_calculate_helium_pui_density.call_args.args

        self.assertEqual(chunk_of_fifty.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS, actual_he_epoch)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        self.assertEqual(expected_fitting_params, passed_in_fitting_params)
        self.assertEqual(mock_helium_inflow_vector, helium_inflow_vector)

        actual_he_epoch, sw_velocity_vector, density_of_neutral_helium_lut, passed_in_fitting_params, helium_inflow_vector = mock_calculate_helium_pui_temperature.call_args.args

        self.assertEqual(chunk_of_fifty.sci_start_time[0] + FIVE_MINUTES_IN_NANOSECONDS, actual_he_epoch)
        np.testing.assert_array_equal([17, 18, 19], sw_velocity_vector)
        self.assertEqual(mock_density_of_neutral_helium_calibration_table, density_of_neutral_helium_lut)
        self.assertEqual(expected_fitting_params, passed_in_fitting_params)
        self.assertEqual(mock_helium_inflow_vector, helium_inflow_vector)

        mock_calculate_ten_minute_velocities.assert_called_with([returned_proton_sw_speed],
                                                                [returned_proton_sw_deflection_angle],
                                                                [returned_proton_sw_clock_angle],
                                                                [SwapiL3Flags.NONE])
        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date",
                                                                 date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 f"imap_swapi_l3a_pui-he"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_pui-he_{start_date_as_str}_{input_version}"),
                                                            ])

        actual_pui_metadata, actual_pui_epoch, actual_pui_cooling_index, actual_pui_ionization_rate, \
            actual_pui_cutoff_speed, actual_pui_background_rate, actual_pui_density, actual_pui_temperature, \
            actual_quality_flags = mock_pickup_ion_data_constructor.call_args.args
        self.assertEqual(expected_pickup_ion_metadata, actual_pui_metadata)
        np.testing.assert_array_equal(np.array([initial_epoch + FIVE_MINUTES_IN_NANOSECONDS]), actual_pui_epoch)
        np.testing.assert_array_equal(np.array([1]), actual_pui_cooling_index)
        np.testing.assert_array_equal(np.array([2]), actual_pui_ionization_rate)
        np.testing.assert_array_equal(np.array([3]), actual_pui_cutoff_speed)
        np.testing.assert_array_equal(np.array([4]), actual_pui_background_rate)
        np.testing.assert_array_equal(np.array([5]), actual_pui_density)
        np.testing.assert_array_equal(np.array([6]), actual_pui_temperature)
        self.assertEqual([0], actual_quality_flags)

        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", "pui-he")

        self.assertEqual(input_file_names, pickup_ion_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), pickup_ion_data, mock_manager)
        self.assertEqual([expected_cdf_path], product)

    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_temperature')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_helium_pui_density')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_pickup_ion_values')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_ten_minute_velocities')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    def test_process_l3a_pui_combines_proton_and_pui_quality_flags(self,
                                                                   mock_parallel_chunk_runner_class,
                                                                   mock_calculate_ten_minute_velocities,
                                                                   mock_calculate_pickup_ion,
                                                                   mock_calculate_helium_pui_density,
                                                                   mock_calculate_helium_pui_temperature):
        initial_epoch = 10
        epoch = np.arange(initial_epoch, initial_epoch + 50)
        energy = np.tile(np.array([15000, 16000, 17000, 18000, 19000]), (50, 1))
        coincidence_count_rate = np.full((50, 5), 5.0)
        coincidence_count_rate_uncertainty = np.full((50, 5), 0.1)
        chunk_of_fifty = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                     coincidence_count_rate_uncertainty)

        runner_quality_flag = SwapiL3Flags.FIT_FAILED
        mock_parallel_chunk_runner_class.return_value.run.return_value = dict(
            proton_sw_speed=np.array([400000.0]),
            proton_sw_clock_angle=np.array([200.0]),
            proton_sw_deflection_angle=np.array([5.0]),
            quality_flags=np.array([runner_quality_flag]),
        )

        ten_min_quality_flag = SwapiL3Flags.STALE_PROTON
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[17, 18, 19]]), np.array([ten_min_quality_flag]))

        pui_fit_quality_flag = SwapiL3Flags.PUI_FIT_MISSING_UNCERTAINTY
        mock_calculate_pickup_ion.return_value = FittingParameters(1, 2, 3, 4, pui_fit_quality_flag)
        mock_calculate_helium_pui_density.return_value = 5
        mock_calculate_helium_pui_temperature.return_value = 6

        input_metadata = InputMetadata('swapi', 'l3a', datetime(2025, 6, 12), datetime(2025, 6, 13), 'v123')
        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        product = swapi_processor.process_l3a_pui(data=chunk_of_fifty, dependencies=Mock())

        # The runner's per-window proton-fit quality flag is passed through to calculate_ten_minute_velocities.
        self.assertEqual([runner_quality_flag],
                         mock_calculate_ten_minute_velocities.call_args.args[3])

        # The output flag is the OR of the per-window proton flag (returned by ten-min) and the per-window PUI fit flag.
        np.testing.assert_array_equal(
            product.quality_flags,
            np.array([ten_min_quality_flag | pui_fit_quality_flag]),
        )

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_proton(self, mock_spicepy,
                                mock_swapi_l3_dependencies_class,
                                mock_chunk_l2_data,
                                mock_parallel_chunk_runner_class,
                                mock_write_cdf,
                                mock_proton_solar_wind_data_constructor,
                                mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 6, 13)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 6, 12)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

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

        runner_result = dict(
            epoch=np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
            proton_sw_speed=np.array([400000.0]),
            proton_sw_speed_uncert=np.array([2.0]),
            proton_sw_speed_sun=np.array([401000.0]),
            proton_sw_speed_sun_uncert=np.array([2.5]),
            proton_sw_temperature=np.array([99000.0]),
            proton_sw_temperature_uncert=np.array([1000.0]),
            proton_sw_density=np.array([4.97]),
            proton_sw_density_uncert=np.array([0.25]),
            proton_sw_clock_angle=np.array([200.0]),
            proton_sw_clock_angle_uncert=np.array([0.25]),
            proton_sw_deflection_angle=np.array([5.0]),
            proton_sw_deflection_angle_uncert=np.array([0.001]),
            proton_sw_bulk_velocity_rtn_sun=np.array([[400.0, 10.0, 5.0]]),
            proton_sw_bulk_velocity_rtn_sun_covariance=np.array([np.eye(3)]),
            proton_sw_bulk_velocity_rtn_sc=np.array([[370.0, 10.0, 5.0]]),
            proton_sw_bulk_velocity_rtn_sc_covariance=np.array([np.eye(3)]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )
        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.return_value = runner_result

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{GEOMETRIC_FACTOR_PUI_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{HYDROGEN_INFLOW_VECTOR_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{HELIUM_INFLOW_VECTOR_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]

        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        expected_proton_metadata = replace(input_metadata, descriptor="proton-sw")
        proton_solar_wind_data.input_metadata = expected_proton_metadata

        input_metadata.descriptor = "proton-sw"

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "06" /
                             f"imap_swapi_l3a_proton-sw_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.return_value = [chunk_of_five]

        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies.data = Mock(energy=energy)
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = swapi_l3a_dependencies

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        swapi_l3a_dependencies.swapi_response.warm_cache.assert_called_once()

        mock_parallel_chunk_runner_class.assert_called_once_with(
            swapi_l3a_dependencies.swapi_response,
            swapi_l3a_dependencies.efficiency_calibration_table,
        )
        mock_runner.run.assert_called_once()
        actual_chunks = mock_runner.run.call_args.args[0]
        self.assertEqual([chunk_of_five], actual_chunks)

        actual_kwargs = mock_proton_solar_wind_data_constructor.call_args.kwargs
        actual_positional = mock_proton_solar_wind_data_constructor.call_args.args
        self.assertEqual(expected_proton_metadata, actual_positional[0])
        for key, expected_val in runner_result.items():
            np.testing.assert_array_equal(expected_val, actual_kwargs[key])

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

    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.calculate_ten_minute_velocities')
    def test_process_l3a_pui_outputs_fill_for_chunks_with_fill(self,
                                                               mock_calculate_ten_minute_velocities,
                                                               mock_parallel_chunk_runner_class):
        instrument = 'swapi'
        end_date = datetime(2025, 6, 13)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 6, 12)
        input_version = "v123"
        initial_epoch = 10

        epoch = np.arange(initial_epoch, initial_epoch + 50)
        energy = np.tile(np.array([15000, 16000, 17000, 18000, 19000]), (50, 1))
        coincidence_count_rate = np.full((50, 5), 5.0)
        coincidence_count_rate[1, 3] = np.nan
        coincidence_count_rate_uncertainty = np.full((50, 5), 0.1)

        chunk_of_fifty = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                     coincidence_count_rate_uncertainty)

        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.return_value = dict(
            proton_sw_speed=np.array([np.nan]),
            proton_sw_clock_angle=np.array([np.nan]),
            proton_sw_deflection_angle=np.array([np.nan]),
            quality_flags=np.array([SwapiL3Flags.NONE]),
        )
        mock_calculate_ten_minute_velocities.return_value = (
            np.array([[np.nan, np.nan, np.nan]]),
            np.array([SwapiL3Flags.NONE]),
        )

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        swapi_processor = SwapiProcessor(Mock(), input_metadata)
        with self.assertLogs(logger):
            product = swapi_processor.process_l3a_pui(data=chunk_of_fifty, dependencies=Mock())

        self.assertIsInstance(product, SwapiL3PickupIonData)
        np.testing.assert_array_equal(product.epoch, initial_epoch + FIVE_MINUTES_IN_NANOSECONDS)

        np.testing.assert_array_equal(nominal_values(product.cooling_index), [np.nan])
        np.testing.assert_array_equal(std_devs(product.cooling_index), [np.nan])

        np.testing.assert_array_equal(nominal_values(product.ionization_rate), [np.nan])
        np.testing.assert_array_equal(std_devs(product.ionization_rate), [np.nan])

        np.testing.assert_array_equal(nominal_values(product.cutoff_speed), [np.nan])
        np.testing.assert_array_equal(std_devs(product.cutoff_speed), [np.nan])

        np.testing.assert_array_equal(nominal_values(product.background_rate), [np.nan])
        np.testing.assert_array_equal(std_devs(product.background_rate), [np.nan])

        np.testing.assert_array_equal(nominal_values(product.density), [np.nan])
        np.testing.assert_array_equal(std_devs(product.density), [np.nan])

        np.testing.assert_array_equal(nominal_values(product.temperature), [np.nan])
        np.testing.assert_array_equal(std_devs(product.temperature), [np.nan])

        np.testing.assert_array_equal(product.quality_flags, [SwapiL3Flags.NONE])

    @patch('imap_l3_processing.utils.ImapAttributeManager')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData')
    @patch('imap_l3_processing.utils.write_cdf')
    @patch('imap_l3_processing.swapi.swapi_processor.ParallelChunkRunner')
    @patch('imap_l3_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_l3_processing.swapi.swapi_processor.SwapiL3ADependencies')
    @patch('imap_l3_processing.processor.spiceypy')
    def test_process_l3a_alpha(self, mock_spicepy,
                               mock_swapi_l3_dependencies_class,
                               mock_chunk_l2_data,
                               mock_parallel_chunk_runner_class,
                               mock_write_cdf,
                               mock_alpha_solar_wind_data_constructor,
                               mock_imap_attribute_manager):
        instrument = 'swapi'
        incoming_data_level = 'l2'
        dependency_start_date = datetime.strftime(datetime(2025, 1, 1), "%Y%m%d")
        version = 'v001'
        end_date = datetime(2025, 8, 29)
        outgoing_data_level = "l3a"
        start_date = datetime(2025, 8, 28)
        input_version = "v123"
        outgoing_version = "123"
        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        mock_spicepy.ktotal.return_value = 0

        initial_epoch = 10
        epoch = np.array([initial_epoch, 11, 12, 13])
        energy = np.array([15000, 16000, 17000, 18000, 19000])
        coincidence_count_rate = np.array(
            [[4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]])
        coincidence_count_rate_uncertainty = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]] * 4)

        chunk_of_five = SwapiL2Data(epoch, energy, coincidence_count_rate,
                                    coincidence_count_rate_uncertainty)

        runner_result = dict(
            epoch=np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]),
            alpha_sw_density=np.array([0.15]),
            alpha_sw_density_uncert=np.array([0.01]),
            alpha_sw_temperature=np.array([400000.0]),
            alpha_sw_temperature_uncert=np.array([2000.0]),
            alpha_sw_velocity_rtn=np.array([[450.0, 5.0, 1.0]]),
            alpha_sw_velocity_covariance_rtn=np.array([np.eye(3)]),
            alpha_sw_delta_v=np.array([12.0]),
            alpha_sw_delta_v_uncert=np.array([1.0]),
            alpha_sw_b_hat_rtn=np.array([[1.0, 0.0, 0.0]]),
            alpha_sw_reference_proton_density=np.array([5.0]),
            alpha_sw_reference_proton_temperature=np.array([99000.0]),
            alpha_sw_reference_proton_velocity_rtn=np.array([[400.0, 5.0, 1.0]]),
            bad_fit_flag=np.array([int(SwapiL3Flags.NONE)]),
        )
        mock_runner = mock_parallel_chunk_runner_class.return_value
        mock_runner.run.return_value = runner_result

        science_input = ScienceInput(
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf')

        input_file_names = [
            f'imap_{instrument}_{incoming_data_level}_{SWAPI_L2_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{EFFICIENCY_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{INSTRUMENT_RESPONSE_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{AZIMUTHAL_TRANSMISSION_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{CENTRAL_EFFECTIVE_AREA_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
            f'imap_{instrument}_{PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
        ]

        ancillary_inputs = [AncillaryInput(file_name) for file_name in input_file_names[1:]]
        dependencies = ProcessingInputCollection(science_input, *ancillary_inputs)

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date, input_version)

        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        expected_alpha_metadata = replace(input_metadata, descriptor="alpha-sw")
        alpha_solar_wind_data.input_metadata = expected_alpha_metadata

        input_metadata.descriptor = "alpha-sw"

        expected_cdf_path = (config["DATA_DIR"] / "imap" / "swapi" / "l3a" / "2025" / "08" /
                             f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_{input_version}.cdf")

        mock_chunk_l2_data.return_value = [chunk_of_five]

        swapi_l3a_dependencies = create_swapi_l3a_dependencies_with_mocks()
        swapi_l3a_dependencies.data = Mock(energy=energy)
        swapi_l3a_dependencies.mag_data = Mock()
        swapi_l3a_dependencies.mag_is_preliminary = False
        mock_swapi_l3_dependencies_class.fetch_dependencies.return_value = swapi_l3a_dependencies

        mock_manager = mock_imap_attribute_manager.return_value

        swapi_processor = SwapiProcessor(dependencies, input_metadata)
        product = swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        swapi_l3a_dependencies.swapi_response.warm_cache.assert_called_once()

        mock_parallel_chunk_runner_class.assert_called_once_with(
            swapi_l3a_dependencies.swapi_response,
            swapi_l3a_dependencies.efficiency_calibration_table,
        )
        mock_runner.run.assert_called_once()
        actual_chunks = mock_runner.run.call_args.args[0]
        self.assertEqual([chunk_of_five], actual_chunks)

        actual_kwargs = mock_alpha_solar_wind_data_constructor.call_args.kwargs
        actual_positional = mock_alpha_solar_wind_data_constructor.call_args.args
        self.assertEqual(expected_alpha_metadata, actual_positional[0])
        for key, expected_val in runner_result.items():
            np.testing.assert_array_equal(expected_val, actual_kwargs[key])

        mock_manager.add_global_attribute.assert_has_calls([
            call("Data_version", outgoing_version),
            call("Generation_date", date.today().strftime("%Y%m%d")),
            call("Logical_source", "imap_swapi_l3a_alpha-sw"),
            call("Logical_file_id", f"imap_swapi_l3a_alpha-sw_{start_date_as_str}_{input_version}"),
        ])
        mock_manager.add_instrument_attrs.assert_called_once_with("swapi", "l3a", "alpha-sw")

        self.assertEqual(input_file_names, alpha_solar_wind_data.parent_file_names)
        mock_write_cdf.assert_called_once_with(str(expected_cdf_path), alpha_solar_wind_data, mock_manager)
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
            f'imap_{instrument}_{incoming_data_level}_{GEOMETRIC_FACTOR_SW_LOOKUP_TABLE_DESCRIPTOR}_{dependency_start_date}_{version}.cdf',
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
        mock_efficiency_table.get_proton_efficiency_for.assert_has_calls(
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
        self.assertEqual(mock_efficiency_table.get_proton_efficiency_for.return_value,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_proton_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[1]))

        self.assertEqual(mock_efficiency_table.get_proton_efficiency_for.return_value,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_alpha_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy, mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates,
                                      nominal_values(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates,
                                      std_devs(mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.get_proton_efficiency_for.return_value,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[2])
        self.assertEqual(mock_geometric_factor_calibration_table,
                         mock_calculate_pui_solar_wind_vdf.call_args_list[0].args[3])

        np.testing.assert_array_equal(energy,
                                      mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[0])
        np.testing.assert_array_equal(nominal_count_rates, nominal_values(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        np.testing.assert_array_equal(std_devs_count_rates, std_devs(
            mock_calculate_combined_solar_wind_differential_flux.call_args_list[0].args[1]))
        self.assertEqual(mock_efficiency_table.get_proton_efficiency_for.return_value,
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


def create_swapi_l3a_dependencies_with_mocks():
    return SwapiL3ADependencies(
        data=Mock(),
        efficiency_calibration_table=Mock(),
        geometric_factor_calibration_table=Mock(),
        instrument_response_calibration_table=Mock(),
        density_of_neutral_helium_calibration_table=Mock(),
        hydrogen_inflow_vector=Mock(),
        helium_inflow_vector=Mock(),
        swapi_response=Mock(),
    )
