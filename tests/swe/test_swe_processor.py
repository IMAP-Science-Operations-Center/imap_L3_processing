import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, call, Mock, sentinel

import numpy as np

from imap_l3_processing.models import MagL1dData, InputMetadata, UpstreamDataDependency
from imap_l3_processing.swe.l3.models import SweL2Data, SwapiL3aProtonData, SweL1bData
from imap_l3_processing.swe.l3.models import SweL3MomentData
from imap_l3_processing.swe.l3.science.moment_calculations import MomentFitResults
from imap_l3_processing.swe.l3.science.moment_calculations import Moments
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.swe_processor import SweProcessor
from tests.test_helpers import NumpyArrayMatcher, build_swe_configuration, create_dataclass_mock


class TestSweProcessor(unittest.TestCase):
    @patch('imap_l3_processing.swe.swe_processor.upload')
    @patch('imap_l3_processing.swe.swe_processor.save_data')
    @patch('imap_l3_processing.swe.swe_processor.SweL3Dependencies.fetch_dependencies')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_products')
    def test_process(self, mock_calculate_products, mock_fetch_dependencies, mock_save_data, mock_upload):
        mock_dependencies = Mock()
        mock_input_metadata = Mock()
        swe_processor = SweProcessor(mock_dependencies, mock_input_metadata)
        swe_processor.process()

        mock_fetch_dependencies.assert_called_once_with(mock_dependencies)
        mock_calculate_products.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_once_with(mock_calculate_products.return_value)
        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch("imap_l3_processing.swe.swe_processor.compute_epoch_delta_in_ns")
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_l3_processing.swe.swe_processor.spice_wrapper')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_pitch_angle_products')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products')
    def test_calculate_products(self, mock_calculate_moment_products, mock_calculate_pitch_angle_products,
                                mock_spice_wrapper, mock_find_breakpoints, mock_average_over_look_directions,
                                mock_compute_epoch_delta_in_ns):
        epochs = datetime.now() + np.arange(2) * timedelta(minutes=1)
        mag_epochs = datetime.now() - timedelta(seconds=15) + np.arange(2) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(2) * timedelta(minutes=.5)
        mock_find_breakpoints.side_effect = [
            (12, 96),
            (16, 86),
        ]
        expected_spacecraft_potential = [12, 16]
        expected_core_halo_breakpoint = [96, 86]

        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([2, 4, 6]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=np.array([])
        )

        expected_corrected_energy_bins = np.array([[2 - 12, 4 - 12, 6 - 12], [2 - 16, 4 - 16, 6 - 16]])

        mag_l1d_data = MagL1dData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
        )
        mock_average_over_look_directions.return_value = np.array([5, 10, 15])

        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_delta=[45, 45, 45],
            energy_bins=[1, 10, 100],
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
        )

        mock_moment_data = create_dataclass_mock(SweL3MomentData)
        mock_calculate_moment_products.return_value = mock_moment_data

        mock_calculate_pitch_angle_products.return_value = (
            sentinel.expected_phase_space_density_by_pitch_angle, sentinel.expected_energy_spectrum,
            sentinel.expected_energy_spectrum_inbound, sentinel.expected_energy_spectrum_outbound,)

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swe_l1b_data = Mock()
        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)

        swe_processor = SweProcessor(swel3_dependency, input_metadata)
        swe_l3_data = swe_processor.calculate_products(swel3_dependency)

        mock_compute_epoch_delta_in_ns.assert_called_once_with(swe_l2_data.acquisition_duration,
                                                               swe_l1b_data.settle_duration)

        mock_spice_wrapper.furnish.assert_called_once()

        self.assertEqual(2, mock_average_over_look_directions.call_count)
        first_average_over_look_directions_call_args = mock_average_over_look_directions.call_args_list[0].args
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[0],
                                      swe_l2_data.phase_space_density[0])
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[1],
                                      swe_config["geometric_fractions"])
        np.testing.assert_array_equal(first_average_over_look_directions_call_args[2],
                                      swe_config["minimum_phase_space_density_value"])

        second_average_over_look_directions_call_args = mock_average_over_look_directions.call_args_list[1].args
        np.testing.assert_array_equal(second_average_over_look_directions_call_args[0],
                                      swe_l2_data.phase_space_density[1])
        np.testing.assert_array_equal(second_average_over_look_directions_call_args[1],
                                      swe_config["geometric_fractions"])
        np.testing.assert_array_equal(second_average_over_look_directions_call_args[2],
                                      swe_config["minimum_phase_space_density_value"])

        mock_find_breakpoints.assert_has_calls([
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 15, 15],
                 [90, 90, 90], swe_config),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 15, 12],
                 [90, 90, 96], swe_config),
        ])

        mock_calculate_moment_products.assert_called_once()
        self.assertEqual(swe_l2_data, mock_calculate_moment_products.call_args[0][0])
        self.assertEqual(swe_l1b_data, mock_calculate_moment_products.call_args[0][1])
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][2], expected_spacecraft_potential)
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][3], expected_core_halo_breakpoint)
        np.testing.assert_array_equal(mock_calculate_moment_products.call_args[0][4], expected_corrected_energy_bins)

        mock_calculate_pitch_angle_products.assert_called_once()
        self.assertEqual(swel3_dependency, mock_calculate_pitch_angle_products.call_args[0][0])
        np.testing.assert_array_equal(mock_calculate_pitch_angle_products.call_args[0][1],
                                      expected_corrected_energy_bins)

        # @formatter:off
        self.assertEqual(swe_l3_data.input_metadata, swe_processor.input_metadata.to_upstream_data_dependency("sci"))
        # pass through from l2
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, mock_compute_epoch_delta_in_ns.return_value)
        # coming from the config
        np.testing.assert_array_equal(swe_l3_data.energy, swe_config["energy_bins"])
        np.testing.assert_array_equal(swe_l3_data.energy_delta_plus, swe_config["energy_delta_plus"])
        np.testing.assert_array_equal(swe_l3_data.energy_delta_minus, swe_config["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.pitch_angle, swe_config["pitch_angle_bins"])
        np.testing.assert_array_equal(swe_l3_data.pitch_angle_delta, swe_config["pitch_angle_delta"])

        # need for both moments and pitch angle
        np.testing.assert_array_equal(swe_l3_data.spacecraft_potential, expected_spacecraft_potential)
        np.testing.assert_array_equal(swe_l3_data.core_halo_breakpoint, expected_core_halo_breakpoint)

        # pitch angle specific
        self.assertEqual(sentinel.expected_phase_space_density_by_pitch_angle, swe_l3_data.phase_space_density_by_pitch_angle)
        self.assertEqual(sentinel.expected_energy_spectrum, swe_l3_data.energy_spectrum)
        self.assertEqual(sentinel.expected_energy_spectrum_inbound, swe_l3_data.energy_spectrum_inbound)
        self.assertEqual(sentinel.expected_energy_spectrum_outbound, swe_l3_data.energy_spectrum_outbound)

        # fit moments
        self.assertEqual(mock_moment_data.core_fit_num_points, swe_l3_data.core_fit_num_points)
        self.assertEqual(mock_moment_data.core_chisq, swe_l3_data.core_chisq)
        self.assertEqual(mock_moment_data.halo_chisq, swe_l3_data.halo_chisq)
        self.assertEqual(mock_moment_data.core_density_fit, swe_l3_data.core_density_fit)
        self.assertEqual(mock_moment_data.halo_density_fit, swe_l3_data.halo_density_fit)
        self.assertEqual(mock_moment_data.core_t_parallel_fit, swe_l3_data.core_t_parallel_fit)
        self.assertEqual(mock_moment_data.halo_t_parallel_fit, swe_l3_data.halo_t_parallel_fit)
        self.assertEqual(mock_moment_data.core_t_perpendicular_fit, swe_l3_data.core_t_perpendicular_fit)
        self.assertEqual(mock_moment_data.halo_t_perpendicular_fit, swe_l3_data.halo_t_perpendicular_fit)
        self.assertEqual(mock_moment_data.core_temperature_phi_rtn_fit, swe_l3_data.core_temperature_phi_rtn_fit)
        self.assertEqual(mock_moment_data.halo_temperature_phi_rtn_fit, swe_l3_data.halo_temperature_phi_rtn_fit)
        self.assertEqual(mock_moment_data.core_temperature_theta_rtn_fit,swe_l3_data.core_temperature_theta_rtn_fit)
        self.assertEqual(mock_moment_data.halo_temperature_theta_rtn_fit,swe_l3_data.halo_temperature_theta_rtn_fit)
        self.assertEqual(mock_moment_data.core_speed_fit, swe_l3_data.core_speed_fit)
        self.assertEqual(mock_moment_data.halo_speed_fit, swe_l3_data.halo_speed_fit)
        self.assertEqual(mock_moment_data.core_velocity_vector_rtn_fit,swe_l3_data.core_velocity_vector_rtn_fit)
        self.assertEqual(mock_moment_data.halo_velocity_vector_rtn_fit,swe_l3_data.halo_velocity_vector_rtn_fit)
        # @formatter:on

    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_l3_processing.swe.swe_processor.calculate_solar_wind_velocity_vector')
    @patch('imap_l3_processing.swe.swe_processor.correct_and_rebin')
    @patch('imap_l3_processing.swe.swe_processor.integrate_distribution_to_get_1d_spectrum')
    @patch('imap_l3_processing.swe.swe_processor.integrate_distribution_to_get_inbound_and_outbound_1d_spectrum')
    @patch('imap_l3_processing.swe.swe_processor.find_closest_neighbor')
    def test_calculate_pitch_angle_products(self, mock_find_closest_neighbor,
                                            mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum,
                                            mock_integrate_distribution_to_get_1d_spectrum,
                                            mock_correct_and_rebin,
                                            mock_calculate_solar_wind_velocity_vector,
                                            mock_find_breakpoints,
                                            mock_average_over_look_directions):
        epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)
        mag_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        spacecraft_potential = np.array([12, 16, 19])
        halo_core = [96, 86, 89]
        energies = np.array([2, 4, 6])

        corrected_energy_bins = energies.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        pitch_angle_bins = [0, 90, 180]

        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=energies,
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=np.array([])
        )

        mag_l1d_data = MagL1dData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
        )
        mock_average_over_look_directions.return_value = np.array([5, 10, 15])
        closest_mag_data = np.arange(9).reshape(3, 3)
        closest_swapi_data = np.arange(8, 17).reshape(3, 3)
        mock_find_closest_neighbor.side_effect = [
            closest_mag_data,
            closest_swapi_data,
        ]

        rebinned_by_pitch = [
            i + np.arange(len(swe_l2_data.energy) * len(pitch_angle_bins)).reshape(len(swe_l2_data.energy),
                                                                                   len(pitch_angle_bins)) for i in
            range(len(epochs))]
        mock_correct_and_rebin.side_effect = rebinned_by_pitch
        integrated_spectrum = np.arange(9).reshape(3, 3) + 11

        mock_integrate_distribution_to_get_1d_spectrum.side_effect = integrated_spectrum

        expected_inbound_spectrum = np.arange(9).reshape(3, 3) + 12
        expected_outbound_spectrum = np.arange(9).reshape(3, 3) + 13
        mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum.side_effect = [
            (expected_inbound_spectrum[0], expected_outbound_spectrum[0]),
            (expected_inbound_spectrum[1], expected_outbound_spectrum[1]),
            (expected_inbound_spectrum[2], expected_outbound_spectrum[2])]

        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_delta=[45, 45, 45],
            energy_bins=[1, 10, 100],
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swel3_dependency = SweL3Dependencies(swe_l2_data, Mock(), mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)

        actual_phase_space_density_by_pitch_angle, actual_energy_spectrum, actual_energy_spectrum_inbound, actual_energy_spectrum_outbound = swe_processor.calculate_pitch_angle_products(
            swel3_dependency, corrected_energy_bins)

        self.assertEqual(3, mock_correct_and_rebin.call_count)
        self.assertEqual(3, mock_integrate_distribution_to_get_1d_spectrum.call_count)
        mock_calculate_solar_wind_velocity_vector.assert_called_once_with(
            swel3_dependency.swapi_l3a_proton_data.proton_sw_speed,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_clock_angle,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_deflection_angle)
        mock_find_closest_neighbor.assert_has_calls([
            call(
                from_epoch=mag_epochs,
                from_data=mag_l1d_data.mag_data,
                to_epoch=swe_l2_data.acquisition_time,
                maximum_distance=np.timedelta64(1, 'm')
            ),
            call(
                from_epoch=swapi_epochs,
                from_data=mock_calculate_solar_wind_velocity_vector.return_value,
                to_epoch=epochs,
                maximum_distance=timedelta(minutes=5)
            )
        ])

        np.testing.assert_array_equal(actual_phase_space_density_by_pitch_angle, rebinned_by_pitch)
        np.testing.assert_array_equal(actual_energy_spectrum, integrated_spectrum)
        np.testing.assert_array_equal(actual_energy_spectrum_inbound, expected_inbound_spectrum)
        np.testing.assert_array_equal(actual_energy_spectrum_outbound, expected_outbound_spectrum)

        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        actual_calls = mock_correct_and_rebin.call_args_list

        expected_calls = [
            call_with_array_matchers(swe_l2_data.phase_space_density[0], swe_l2_data.energy - 12, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0],
                                     closest_mag_data[0], closest_swapi_data[0], swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[1], swe_l2_data.energy - 16, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1],
                                     closest_mag_data[1], closest_swapi_data[1], swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[2], swe_l2_data.energy - 19, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2],
                                     closest_mag_data[2], closest_swapi_data[2], swe_config)
        ]
        self.assertEqual(actual_calls, expected_calls)
        mock_integrate_distribution_to_get_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch[0], swe_config),
            call(rebinned_by_pitch[1], swe_config),
            call(rebinned_by_pitch[2], swe_config)
        ])
        mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch[0], swe_config),
            call(rebinned_by_pitch[1], swe_config),
            call(rebinned_by_pitch[2], swe_config)
        ])

    @patch("imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products")
    def test_calculate_pitch_angle_products_makes_nan_if_no_mag_close_enough(self, _):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_epochs = np.array([datetime(2025, 3, 6, 0, 1, 30)])
        swapi_epochs = np.array([datetime(2025, 3, 6)])

        pitch_angle_bins = [70, 100, 130]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 20,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_duration=np.full((num_epochs, num_energies, 5), 80000)
        )

        swe_l1b_data = SweL1bData(
            epoch=epochs,
            count_rates=Mock(),
            settle_duration=np.full((num_epochs, 3), 333)
        )

        mag_l1d_data = MagL1dData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.full(len(swapi_epochs), 400),
            proton_sw_clock_angle=np.full(len(swapi_epochs), 0),
            proton_sw_deflection_angle=np.full(len(swapi_epochs), 0),
        )
        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        energy_bins = [8, 10, 13]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_delta=[15, 15, 15],
            energy_bins=energy_bins,
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
            in_vs_out_energy_index=len(energy_bins) - 1
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_products(swel3_dependency)

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_by_pitch_angle,
                                      np.full((len(epochs), len(energy_bins), len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum, np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_inbound,
                                      np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_outbound,
                                      np.full((len(epochs), len(energy_bins)), np.nan))

    @patch("imap_l3_processing.swe.swe_processor.SweProcessor.calculate_moment_products")
    def test_calculate_pitch_angle_products_without_mocks(self, _):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_epochs = np.array([datetime(2025, 3, 6, 0, 0, 30)])
        swapi_epochs = np.array([datetime(2025, 3, 6)])

        pitch_angle_bins = [70, 100, 130]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 20,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_duration=np.full((num_epochs, num_energies, 5), 80000)
        )

        mag_l1d_data = MagL1dData(
            epoch=mag_epochs,
            mag_data=np.arange(7, 22).reshape(5, 3).repeat(2, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3aProtonData(
            epoch=swapi_epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 10),
            proton_sw_speed=np.full(len(swapi_epochs), 400),
            proton_sw_clock_angle=np.full(len(swapi_epochs), 0),
            proton_sw_deflection_angle=np.full(len(swapi_epochs), 0),
        )
        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        energy_bins = [8, 10, 13]
        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_delta=[15, 15, 15],
            energy_bins=energy_bins,
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
            max_swapi_offset_in_minutes=5,
            max_mag_offset_in_minutes=1,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
            in_vs_out_energy_index=len(energy_bins) - 1
        )

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swe_l1b_data = SweL1bData(
            epoch=epochs,
            count_rates=Mock(),
            settle_duration=np.full((num_epochs, 3), 333)
        )

        swel3_dependency = SweL3Dependencies(swe_l2_data, swe_l1b_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_products(swel3_dependency)

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_allclose(swe_l3_data.phase_space_density_by_pitch_angle,
                                   np.array([[[np.nan, 194.772034, 270.312835],
                                              [211.672665, 273.136802, 363.195552],
                                              [313.860051, 400.23009, np.nan]]]))
        np.testing.assert_allclose(swe_l3_data.epoch_delta, np.array([1807492500]))
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_allclose(swe_l3_data.energy_spectrum, np.array([[104.427758, 195.333344, 180.401159]]))
        np.testing.assert_allclose(swe_l3_data.energy_spectrum_inbound, np.array([[0, 104.147588, 154.42602]]))
        np.testing.assert_allclose(swe_l3_data.energy_spectrum_outbound,
                                   np.array([[208.855516, 286.519101, 206.376298]]))

    @patch('imap_l3_processing.swe.swe_processor.halo_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.core_fit_moments_retrying_on_failure')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    def test_calculate_moment_products(self, mock_rotate_temperature,
                                       mock_rotate_dps_vector_to_rtn,
                                       mock_calculate_velocity_in_dsp_frame_km_s,
                                       mock_compute_maxwellian_weight_factors,
                                       mock_core_fit_moments_retrying_on_failure: Mock,
                                       mock_halo_fit_moments_retrying_on_failure: Mock):
        epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)

        swe_l2_data = SweL2Data(
            epoch=epochs,
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
            acquisition_duration=[sentinel.acquisition_duration_1, sentinel.acquisition_duration_2,
                                  sentinel.acquisition_duration_3],
        )
        swe_l1_data = SweL1bData(epoch=epochs,
                                 count_rates=[sentinel.l1b_count_rates_1, sentinel.l1b_count_rates_2,
                                              sentinel.count_rates_3],
                                 settle_duration=Mock())

        spacecraft_potential = np.array([12, 14, 16])
        core_halo_breakpoint = np.array([96, 54, 103])

        corrected_energy_bins = swe_l2_data.energy.reshape(1, -1) - spacecraft_potential.reshape(-1, 1)

        velocity_in_dsp_frame_km_s = [
            np.full(shape=(24, 30, 7, 3), fill_value=1),
            np.full(shape=(24, 30, 7, 3), fill_value=2),
            np.full(shape=(24, 30, 7, 3), fill_value=3)
        ]

        mock_calculate_velocity_in_dsp_frame_km_s.side_effect = velocity_in_dsp_frame_km_s

        maxwellian_weight_factors = [np.full(shape=(24, 30, 7), fill_value=1),
                                     np.full(shape=(24, 30, 7), fill_value=2),
                                     np.full(shape=(24, 30, 7), fill_value=3)]

        mock_compute_maxwellian_weight_factors.side_effect = maxwellian_weight_factors

        core_moments1 = create_dataclass_mock(Moments, density=15, velocity_x=5, velocity_y=6, velocity_z=7,
                                              t_parallel=100, t_perpendicular=200)
        halo_moments1 = create_dataclass_mock(Moments, density=22, velocity_x=5, velocity_y=6,
                                              velocity_z=7, t_parallel=101, t_perpendicular=201)
        core_moments2 = create_dataclass_mock(Moments, density=12, velocity_x=8, velocity_y=9, velocity_z=10,
                                              t_parallel=102, t_perpendicular=202)
        halo_moments2 = create_dataclass_mock(Moments, density=19, velocity_x=8, velocity_y=9,
                                              velocity_z=10, t_parallel=103, t_perpendicular=203)

        core_moment_fit_results_1 = MomentFitResults(moments=core_moments1, chisq=4730912.0, number_of_points=10)
        core_moment_fit_results_2 = MomentFitResults(moments=core_moments2, chisq=4705988.0, number_of_points=11)
        halo_moment_fit_results_1 = MomentFitResults(moments=halo_moments1, chisq=3412.0, number_of_points=12)
        halo_moment_fit_results_2 = MomentFitResults(moments=halo_moments2, chisq=3214.0, number_of_points=13)

        mock_core_fit_moments_retrying_on_failure.side_effect = [core_moment_fit_results_1, core_moment_fit_results_2,
                                                                 None]
        mock_halo_fit_moments_retrying_on_failure.side_effect = [halo_moment_fit_results_1, halo_moment_fit_results_2,
                                                                 None]

        core_rtn_theta_1, core_rtn_phi_1 = (62, 190)
        halo_rtn_theta_1, halo_rtn_phi_1 = (54, 222)
        core_rtn_theta_2, core_rtn_phi_2 = (63, 191)
        halo_rtn_theta_2, halo_rtn_phi_2 = (55, 223)
        mock_rotate_temperature.side_effect = [
            (core_rtn_theta_1, core_rtn_phi_1),
            (halo_rtn_theta_1, halo_rtn_phi_1),
            (core_rtn_theta_2, core_rtn_phi_2),
            (halo_rtn_theta_2, halo_rtn_phi_2),
        ]

        core_velocity_1 = [0, 0, 1]
        halo_velocity_1 = [0, 1, 0]
        core_velocity_2 = [0, 1, 1]
        halo_velocity_2 = [1, 0, 0]
        mock_rotate_dps_vector_to_rtn.side_effect = [
            core_velocity_1,
            halo_velocity_1,
            core_velocity_2,
            halo_velocity_2
        ]

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_moment_data = swe_processor.calculate_moment_products(swe_l2_data, swe_l1_data, spacecraft_potential,
                                                                  core_halo_breakpoint,
                                                                  corrected_energy_bins)

        calculate_velocity_call_1 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[0]
        np.testing.assert_array_equal(corrected_energy_bins[0], calculate_velocity_call_1.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[0], calculate_velocity_call_1.args[2])

        calculate_velocity_call_2 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1], calculate_velocity_call_2.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[1], calculate_velocity_call_2.args[2])

        calculate_velocity_call_3 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2], calculate_velocity_call_3.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[2], calculate_velocity_call_3.args[2])

        mock_compute_maxwellian_weight_factors.assert_has_calls(
            [call(sentinel.l1b_count_rates_1, sentinel.acquisition_duration_1),
             call(sentinel.l1b_count_rates_2, sentinel.acquisition_duration_2)])

        self.assertEqual(3, mock_core_fit_moments_retrying_on_failure.call_count)
        self.assertEqual(3, mock_halo_fit_moments_retrying_on_failure.call_count)

        core_fit_moments_call_1 = mock_core_fit_moments_retrying_on_failure.mock_calls[0]
        np.testing.assert_array_equal(core_fit_moments_call_1.args[0], corrected_energy_bins[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0],
                                      core_fit_moments_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], core_fit_moments_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      core_fit_moments_call_1.args[3])
        self.assertEqual(2, core_fit_moments_call_1.args[4])
        self.assertEqual(6, core_fit_moments_call_1.args[5])
        np.testing.assert_array_equal(core_fit_moments_call_1.args[6], [100, 100, 100])

        halo_fit_moments_call_1 = mock_halo_fit_moments_retrying_on_failure.mock_calls[0]
        np.testing.assert_array_equal(corrected_energy_bins[0],
                                      halo_fit_moments_call_1.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0],
                                      halo_fit_moments_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], halo_fit_moments_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      halo_fit_moments_call_1.args[3])
        self.assertEqual(6, halo_fit_moments_call_1.args[4])
        self.assertEqual(10, halo_fit_moments_call_1.args[5])
        np.testing.assert_array_equal(halo_fit_moments_call_1.args[6], [25, 25, 25])
        self.assertEqual(spacecraft_potential[0], halo_fit_moments_call_1.args[7])
        self.assertEqual(core_halo_breakpoint[0], halo_fit_moments_call_1.args[8])

        core_fit_moments_2 = mock_core_fit_moments_retrying_on_failure.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1],
                                      core_fit_moments_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1],
                                      core_fit_moments_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], core_fit_moments_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      core_fit_moments_2.args[3])
        self.assertEqual(3, core_fit_moments_2.args[4])
        self.assertEqual(5, core_fit_moments_2.args[5])
        np.testing.assert_array_equal(core_fit_moments_2.args[6], [100, 100, core_moments1.density])

        halo_fit_moments_2 = mock_halo_fit_moments_retrying_on_failure.mock_calls[1]
        np.testing.assert_array_equal(corrected_energy_bins[1],
                                      halo_fit_moments_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1],
                                      halo_fit_moments_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], halo_fit_moments_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      halo_fit_moments_2.args[3])
        self.assertEqual(5, halo_fit_moments_2.args[4])
        self.assertEqual(10, halo_fit_moments_2.args[5])
        np.testing.assert_array_equal(halo_fit_moments_2.args[6], [25, 25, halo_moments1.density])
        self.assertEqual(spacecraft_potential[1], halo_fit_moments_2.args[7])
        self.assertEqual(core_halo_breakpoint[1], halo_fit_moments_2.args[8])

        core_fit_moments_3 = mock_core_fit_moments_retrying_on_failure.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2],
                                      core_fit_moments_3.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[2],
                                      core_fit_moments_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[2], core_fit_moments_3.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[2],
                                      core_fit_moments_3.args[3])
        self.assertEqual(4, core_fit_moments_3.args[4])
        self.assertEqual(7, core_fit_moments_3.args[5])
        np.testing.assert_array_equal(core_fit_moments_3.args[6], [100, core_moments1.density, core_moments2.density])

        halo_fit_moments_3 = mock_halo_fit_moments_retrying_on_failure.mock_calls[2]
        np.testing.assert_array_equal(corrected_energy_bins[2],
                                      halo_fit_moments_3.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[2],
                                      halo_fit_moments_3.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[2], halo_fit_moments_3.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[2],
                                      halo_fit_moments_3.args[3])
        self.assertEqual(7, halo_fit_moments_3.args[4])
        self.assertEqual(10, halo_fit_moments_3.args[5])
        np.testing.assert_array_equal(halo_fit_moments_3.args[6], [25, halo_moments1.density, halo_moments2.density])
        self.assertEqual(spacecraft_potential[2], halo_fit_moments_3.args[7])
        self.assertEqual(core_halo_breakpoint[2], halo_fit_moments_3.args[8])

        self.assertEqual(4, mock_rotate_dps_vector_to_rtn.call_count)

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[0].args[0])
        np.testing.assert_array_equal(
            np.array([core_moments1.velocity_x, core_moments1.velocity_y, core_moments1.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[0].args[1])

        self.assertEqual(epochs[0], mock_rotate_dps_vector_to_rtn.call_args_list[1].args[0])
        np.testing.assert_array_equal(
            np.array([halo_moments1.velocity_x, halo_moments1.velocity_y, halo_moments1.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[1].args[1])

        self.assertEqual(epochs[1], mock_rotate_dps_vector_to_rtn.call_args_list[2].args[0])
        np.testing.assert_array_equal(
            np.array([core_moments2.velocity_x, core_moments2.velocity_y, core_moments2.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[2].args[1])

        self.assertEqual(epochs[1], mock_rotate_dps_vector_to_rtn.call_args_list[2].args[0])
        np.testing.assert_array_equal(
            np.array([halo_moments2.velocity_x, halo_moments2.velocity_y, halo_moments2.velocity_z]),
            mock_rotate_dps_vector_to_rtn.call_args_list[2].args[1])

        self.assertEqual(4, mock_rotate_temperature.call_count)
        mock_rotate_temperature.assert_has_calls(
            [call(epochs[0], core_moments1.alpha, core_moments1.beta),
             call(epochs[0], halo_moments1.alpha, halo_moments1.beta),
             call(epochs[1], core_moments2.alpha, core_moments2.beta),
             call(epochs[1], halo_moments2.alpha, halo_moments2.beta), ])

        # @formatter:off
        self.assertIsInstance(swe_moment_data, SweL3MomentData)
        np.testing.assert_array_equal(swe_moment_data.core_fit_num_points,[core_moment_fit_results_1.number_of_points,core_moment_fit_results_2.number_of_points, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_chisq,[core_moment_fit_results_1.chisq, core_moment_fit_results_2.chisq, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_chisq,[halo_moment_fit_results_1.chisq, halo_moment_fit_results_2.chisq, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_density_fit,[core_moments1.density, core_moments2.density, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_density_fit,[halo_moments1.density, halo_moments2.density, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_t_parallel_fit,[core_moments1.t_parallel, core_moments2.t_parallel, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_t_parallel_fit,[halo_moments1.t_parallel, halo_moments2.t_parallel, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_t_perpendicular_fit,[core_moments1.t_perpendicular, core_moments2.t_perpendicular, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_t_perpendicular_fit,[halo_moments1.t_perpendicular, halo_moments2.t_perpendicular, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_temperature_phi_rtn_fit,[core_rtn_phi_1, core_rtn_phi_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_phi_rtn_fit,[halo_rtn_phi_1, halo_rtn_phi_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_temperature_theta_rtn_fit,[core_rtn_theta_1, core_rtn_theta_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_temperature_theta_rtn_fit,[halo_rtn_theta_1, halo_rtn_theta_2, np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_speed_fit,[np.linalg.norm(core_velocity_1), np.linalg.norm(core_velocity_2), np.nan])
        np.testing.assert_array_equal(swe_moment_data.halo_speed_fit,[np.linalg.norm(halo_velocity_1), np.linalg.norm(halo_velocity_2), np.nan])
        np.testing.assert_array_equal(swe_moment_data.core_velocity_vector_rtn_fit,[core_velocity_1, core_velocity_2, [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(swe_moment_data.halo_velocity_vector_rtn_fit,[halo_velocity_1, halo_velocity_2, [np.nan, np.nan, np.nan]])
        # @formatter:on
