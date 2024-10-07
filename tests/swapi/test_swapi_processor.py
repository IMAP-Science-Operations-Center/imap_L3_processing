import os
import shutil
from datetime import datetime, timedelta, date
from unittest import TestCase
from unittest.mock import patch, sentinel, call

import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray, nominal_values, std_devs

from imap_processing.constants import TEMP_CDF_FOLDER_PATH, THIRTY_SECONDS_IN_NANOSECONDS, FIVE_MINUTES_IN_NANOSECONDS
from imap_processing.models import UpstreamDataDependency, InputMetadata
from imap_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.swapi.l3a.swapi_l3a_dependencies import SWAPI_L2_DESCRIPTOR
from imap_processing.swapi.l3b.science.calculate_solar_wind_vdf import DeltaMinusPlus
from imap_processing.swapi.swapi_processor import SwapiProcessor


class TestSwapiProcessor(TestCase):
    def setUp(self) -> None:
        self.temp_directory = f"{TEMP_CDF_FOLDER_PATH}"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)
        self.mock_imap_patcher = patch('imap_processing.swapi.swapi_processor.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
        self.mock_imap_patcher.stop()

    @patch('imap_processing.utils.ImapAttributeManager')
    @patch('imap_processing.swapi.swapi_processor.SwapiL3AlphaSolarWindData')
    @patch('imap_processing.swapi.swapi_processor.SwapiL3ProtonSolarWindData')
    @patch('imap_processing.utils.write_cdf')
    @patch('imap_processing.utils.uuid')
    @patch('imap_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_processing.swapi.swapi_processor.read_l2_swapi_data')
    @patch('imap_processing.swapi.swapi_processor.calculate_proton_solar_wind_speed')
    @patch('imap_processing.swapi.swapi_processor.calculate_alpha_solar_wind_speed')
    @patch('imap_processing.swapi.swapi_processor.calculate_proton_solar_wind_temperature_and_density')
    @patch(
        'imap_processing.swapi.swapi_processor.calculate_alpha_solar_wind_temperature_and_density_for_combined_sweeps')
    @patch('imap_processing.swapi.swapi_processor.calculate_clock_angle')
    @patch('imap_processing.swapi.swapi_processor.calculate_deflection_angle')
    @patch('imap_processing.swapi.swapi_processor.SwapiL3ADependencies')
    def test_processor(self, mock_swapi_l3_dependencies_class,
                       mock_calculate_deflection_angle,
                       mock_calculate_clock_angle,
                       mock_alpha_calculate_temperature_and_density,
                       mock_proton_calculate_temperature_and_density,
                       mock_calculate_alpha_solar_wind_speed,
                       mock_calculate_proton_solar_wind_speed,
                       mock_read_l2_swapi_data, mock_chunk_l2_data, mock_uuid, mock_write_cdf,
                       mock_proton_solar_wind_data_constructor, mock_alpha_solar_wind_data_constructor,
                       mock_imap_attribute_manager):
        mock_uuid_value = 123
        mock_uuid.uuid4.return_value = mock_uuid_value

        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = SWAPI_L2_DESCRIPTOR
        end_date = datetime.now()
        version = 'f'
        outgoing_data_level = "l3a"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

        start_date_as_str = datetime.strftime(start_date, "%Y%m%d")

        returned_proton_sw_speed = ufloat(400000, 2)
        mock_calculate_proton_solar_wind_speed.return_value = (
            returned_proton_sw_speed, sentinel.a, sentinel.phi, sentinel.b)

        returned_alpha_temperature = ufloat(400000, 2000)
        returned_alpha_density = ufloat(0.15, 0.01)
        mock_alpha_calculate_temperature_and_density.return_value = (returned_alpha_temperature, returned_alpha_density)

        returned_alpha_speed = ufloat(450000, 1000)
        mock_calculate_alpha_solar_wind_speed.return_value = ufloat(450000, 1000)

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
        spin_angles = np.array([[24, 25, 26, 27, 28], [29, 30, 31, 32, 33], [34, 35, 36, 37, 38], [39, 40, 41, 42, 43]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_chunk_l2_data.return_value = [
            SwapiL2Data(epoch, energy, coincidence_count_rate, spin_angles, coincidence_count_rate_uncertainty)]

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)

        proton_solar_wind_data = mock_proton_solar_wind_data_constructor.return_value
        expected_proton_metadata = input_metadata.to_upstream_data_dependency("proton-sw")
        proton_solar_wind_data.input_metadata = expected_proton_metadata

        alpha_solar_wind_data = mock_alpha_solar_wind_data_constructor.return_value
        expected_alpha_metadata = input_metadata.to_upstream_data_dependency("alpha-sw")
        alpha_solar_wind_data.input_metadata = expected_alpha_metadata

        mock_manager = mock_imap_attribute_manager.return_value

        dependencies = [
            UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                   version, descriptor),
        ]

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        swapi_processor.process()

        mock_swapi_l3_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_proton_temperature_density_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.proton_temperature_density_calibration_table
        mock_alpha_temperature_density_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.alpha_temperature_density_calibration_table
        mock_clock_angle_and_flow_deflection_calibration_table = mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.clock_angle_and_flow_deflection_calibration_table

        mock_read_l2_swapi_data.assert_called_once_with(
            mock_swapi_l3_dependencies_class.fetch_dependencies.return_value.data)

        mock_chunk_l2_data.assert_called_with(mock_read_l2_swapi_data.return_value, 5)

        expected_count_rate_with_uncertainties = uarray(coincidence_count_rate, coincidence_count_rate_uncertainty)
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_proton_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(spin_angles, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[1])
        np.testing.assert_array_equal(energy, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[2])
        np.testing.assert_array_equal(epoch, mock_calculate_proton_solar_wind_speed.call_args_list[0].args[3])

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
        self.assert_ufloat_equal(ufloat(0.01, 1.0),
                                 mock_proton_calculate_temperature_and_density.call_args_list[0].args[2])
        self.assertEqual(sentinel.phi, mock_proton_calculate_temperature_and_density.call_args_list[0].args[3])
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_proton_calculate_temperature_and_density.call_args_list[0].args[4]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_proton_calculate_temperature_and_density.call_args_list[0].args[4]))
        np.testing.assert_array_equal(energy, mock_proton_calculate_temperature_and_density.call_args_list[0].args[5])

        self.assertEqual(mock_alpha_temperature_density_calibration_table,
                         mock_alpha_calculate_temperature_and_density.call_args_list[0].args[0])
        self.assert_ufloat_equal(returned_alpha_speed,
                                 mock_alpha_calculate_temperature_and_density.call_args_list[0].args[1])
        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(
                                          mock_alpha_calculate_temperature_and_density.call_args_list[0].args[2]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_alpha_calculate_temperature_and_density.call_args_list[0].args[2]))
        np.testing.assert_array_equal(energy, mock_alpha_calculate_temperature_and_density.call_args_list[0].args[3])

        np.testing.assert_array_equal(nominal_values(expected_count_rate_with_uncertainties),
                                      nominal_values(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(std_devs(expected_count_rate_with_uncertainties),
                                      std_devs(mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[0]))
        np.testing.assert_array_equal(energy, mock_calculate_alpha_solar_wind_speed.call_args_list[0].args[1])

        (actual_proton_metadata, actual_proton_epoch, actual_proton_sw_speed, actual_proton_sw_temperature,
         actual_proton_sw_density, actual_proton_sw_clock_angle,
         actual_proton_sw_deflection_angle) = mock_proton_solar_wind_data_constructor.call_args.args

        self.assertEqual(expected_proton_metadata, actual_proton_metadata)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), actual_proton_epoch,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_speed]), actual_proton_sw_speed, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_temp]), actual_proton_sw_temperature, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_density]), actual_proton_sw_density, strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_clock_angle]), actual_proton_sw_clock_angle,
                                      strict=True)
        np.testing.assert_array_equal(np.array([returned_proton_sw_deflection_angle]),
                                      actual_proton_sw_deflection_angle,
                                      strict=True)

        proton_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_proton-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        alpha_cdf_path = f"{self.temp_directory}/imap_swapi_l3a_alpha-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345.cdf"
        mock_manager.add_global_attribute.assert_has_calls([call("Data_version", outgoing_version),
                                                            call("Generation_date", date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 "imap_swapi_l3a_proton-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_proton-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345"),
                                                            call("Data_version", outgoing_version),
                                                            call("Generation_date", date.today().strftime("%Y%m%d")),
                                                            call("Logical_source",
                                                                 "imap_swapi_l3a_alpha-sw"),
                                                            call("Logical_file_id",
                                                                 f"imap_swapi_l3a_alpha-sw-fake-menlo-{mock_uuid_value}_{start_date_as_str}_12345"),
                                                            ])

        actual_alpha_metadata, actual_alpha_epoch, actual_alpha_sw_speed, actual_alpha_sw_temperature, actual_alpha_sw_density = mock_alpha_solar_wind_data_constructor.call_args.args
        expected_alpha_metadata = input_metadata.to_upstream_data_dependency("alpha-sw")
        self.assertEqual(expected_alpha_metadata, actual_alpha_metadata)

        np.testing.assert_array_equal(np.array([initial_epoch + THIRTY_SECONDS_IN_NANOSECONDS]), actual_alpha_epoch)
        np.testing.assert_array_equal(np.array([mock_calculate_alpha_solar_wind_speed.return_value]),
                                      actual_alpha_sw_speed)
        np.testing.assert_array_equal(np.array([mock_alpha_calculate_temperature_and_density.return_value[0]]),
                                      actual_alpha_sw_temperature)
        np.testing.assert_array_equal(np.array([mock_alpha_calculate_temperature_and_density.return_value[1]]),
                                      actual_alpha_sw_density)

        mock_manager.add_instrument_attrs.assert_has_calls([call("swapi", "l3a"), call("swapi", "l3a")])
        mock_write_cdf.assert_has_calls([
            call(proton_cdf_path, proton_solar_wind_data, mock_manager),
            call(alpha_cdf_path, alpha_solar_wind_data, mock_manager)
        ])
        self.mock_imap_api.upload.assert_has_calls([call(proton_cdf_path), call(alpha_cdf_path)])

    @patch('imap_processing.swapi.swapi_processor.imap_data_access')
    @patch('imap_processing.swapi.swapi_processor.calculate_delta_minus_plus')
    @patch('imap_processing.swapi.swapi_processor.save_data')
    @patch('imap_processing.swapi.swapi_processor.SwapiL3BCombinedVDF')
    @patch('imap_processing.swapi.swapi_processor.calculate_combined_sweeps')
    @patch('imap_processing.utils.uuid')
    @patch('imap_processing.swapi.swapi_processor.chunk_l2_data')
    @patch('imap_processing.swapi.swapi_processor.read_l2_swapi_data')
    @patch('imap_processing.swapi.swapi_processor.SwapiL3BDependencies')
    @patch('imap_processing.swapi.swapi_processor.calculate_alpha_solar_wind_vdf')
    @patch('imap_processing.swapi.swapi_processor.calculate_proton_solar_wind_vdf')
    @patch('imap_processing.swapi.swapi_processor.calculate_pui_solar_wind_vdf')
    @patch('imap_processing.swapi.swapi_processor.calculate_combined_solar_wind_differential_flux')
    def test_process_l3b(self, mock_calculate_combined_solar_wind_differential_flux, mock_calculate_pui_solar_wind_vdf,
                         mock_calculate_proton_solar_wind_vdf,
                         mock_calculate_alpha_solar_wind_vdf,
                         mock_swapi_l3b_dependencies_class,
                         mock_read_l2_swapi_data, mock_chunk_l2_data, mock_uuid,
                         mock_calculate_combined_sweeps, mock_combined_vdf_data,
                         mock_save_data,
                         mock_calculate_delta_minus_plus,
                         mock_imap_data_access):
        mock_uuid_value = 123
        mock_uuid.uuid4.return_value = mock_uuid_value

        instrument = 'swapi'
        incoming_data_level = 'l2'
        descriptor = SWAPI_L2_DESCRIPTOR
        end_date = datetime.now()
        version = 'f'
        outgoing_data_level = "l3b"
        start_date = datetime.now() - timedelta(days=1)
        outgoing_version = "12345"

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

        spin_angles = np.array([[24, 25, 26, 27, 28], [29, 30, 31, 32, 33], [34, 35, 36, 37, 38], [39, 40, 41, 42, 43]])
        coincidence_count_rate_uncertainty = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
             [0.1, 0.2, 0.3, 0.4, 0.5]])
        average_coincident_count_rate_uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5]

        first_chunk_initial_epoch = 10
        first_l2_data_chunk = SwapiL2Data(np.array([first_chunk_initial_epoch, 11, 12, 13]), sentinel.energies,
                                          coincidence_count_rate, spin_angles,
                                          coincidence_count_rate_uncertainty)

        second_chunk_initial_epoch = 60
        second_l2_data_chunk = SwapiL2Data(np.array([second_chunk_initial_epoch, 11, 12, 13]), sentinel.energies,
                                           coincidence_count_rate, spin_angles,
                                           coincidence_count_rate_uncertainty)

        mock_chunk_l2_data.return_value = [first_l2_data_chunk, second_l2_data_chunk]

        mock_calculate_combined_sweeps.return_value = [
            uarray(average_coincident_count_rates, average_coincident_count_rate_uncertainties), energy]

        input_metadata = InputMetadata(instrument, outgoing_data_level, start_date, end_date,
                                       outgoing_version)
        dependencies = [
            UpstreamDataDependency(instrument, incoming_data_level, start_date, end_date,
                                   version, descriptor),
        ]

        swapi_processor = SwapiProcessor(
            dependencies, input_metadata)
        swapi_processor.process()

        mock_swapi_l3b_dependencies_class.fetch_dependencies.assert_called_once_with(dependencies)

        mock_geometric_factor_calibration_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.geometric_factor_calibration_table
        mock_efficiency_table = mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.efficiency_calibration_table

        mock_read_l2_swapi_data.assert_called_once_with(
            mock_swapi_l3b_dependencies_class.fetch_dependencies.return_value.data)

        mock_chunk_l2_data.assert_called_with(mock_read_l2_swapi_data.return_value, 50)

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

        expected_combined_metadata = input_metadata.to_upstream_data_dependency("combined")
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

        mock_save_data.assert_called_once_with(mock_combined_vdf_data.return_value)
        mock_imap_data_access.upload.assert_called_with(mock_save_data.return_value)

    def assert_ufloat_equal(self, expected_ufloat, actual_ufloat):
        self.assertEqual(expected_ufloat.n, actual_ufloat.n)
        self.assertEqual(expected_ufloat.s, actual_ufloat.s)
