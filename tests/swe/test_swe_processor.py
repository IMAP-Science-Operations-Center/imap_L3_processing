import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, call, Mock

import numpy as np

from imap_processing.models import MagL1dData, InputMetadata, UpstreamDataDependency
from imap_processing.swe.l3.models import SweL2Data, SweConfiguration, SwapiL3aProtonData
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_processing.swe.swe_processor import SweProcessor
from tests.test_helpers import NumpyArrayMatcher, build_swe_configuration


class TestSweProcessor(unittest.TestCase):
    @patch('imap_processing.swe.swe_processor.upload')
    @patch('imap_processing.swe.swe_processor.save_data')
    @patch('imap_processing.swe.swe_processor.SweL3Dependencies.fetch_dependencies')
    @patch('imap_processing.swe.swe_processor.SweProcessor.calculate_pitch_angle_products')
    def test_process(self, mock_calculate_pitch_angle_products, mock_fetch_dependencies, mock_save_data, mock_upload):
        mock_dependencies = Mock()
        mock_input_metadata = Mock()
        swe_processor = SweProcessor(mock_dependencies, mock_input_metadata)
        swe_processor.process()

        mock_fetch_dependencies.assert_called_once_with(mock_dependencies)
        mock_calculate_pitch_angle_products.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_once_with(mock_calculate_pitch_angle_products.return_value)
        mock_upload.assert_called_once_with(mock_save_data.return_value)

    @patch('imap_processing.swe.swe_processor.average_flux')
    @patch('imap_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_processing.swe.swe_processor.calculate_solar_wind_velocity_vector')
    @patch('imap_processing.swe.swe_processor.correct_and_rebin')
    @patch('imap_processing.swe.swe_processor.integrate_distribution_to_get_1d_spectrum')
    @patch('imap_processing.swe.swe_processor.integrate_distribution_to_get_inbound_and_outbound_1d_spectrum')
    @patch('imap_processing.swe.swe_processor.find_closest_neighbor')
    def test_calculate_pitch_angle_products(self, mock_find_closest_neighbor,
                                            mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum,
                                            mock_integrate_distribution_to_get_1d_spectrum,
                                            mock_correct_and_rebin,
                                            mock_calculate_solar_wind_velocity_vector,
                                            mock_find_breakpoints,
                                            mock_average_flux):
        epochs = datetime.now() + np.arange(3) * timedelta(minutes=1)
        mag_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        mock_find_breakpoints.side_effect = [
            (12, 96),
            (16, 86),
            (19, 89),
        ]
        pitch_angle_bins = [0, 90, 180]

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 3),
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([2, 4, 6]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
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
        mock_average_flux.return_value = np.array([5, 10, 15])
        closest_mag_data = np.arange(9).reshape(3, 3)
        closest_swapi_data = np.arange(8, 17).reshape(3, 3)
        mock_find_closest_neighbor.side_effect = [
            closest_mag_data,
            closest_swapi_data,
        ]

        rebinned_by_pitch = [
            i + np.arange(len(swe_l2_data.energy) * len(pitch_angle_bins)).reshape(len(swe_l2_data.energy),
                                                                                   len(pitch_angle_bins)) for i in
            range(2 * len(epochs))]
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
        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_pitch_angle_products(swel3_dependency)

        self.assertEqual(3, mock_average_flux.call_count)
        self.assertEqual(3, mock_find_breakpoints.call_count)
        self.assertEqual(6, mock_correct_and_rebin.call_count)
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
        mock_average_flux.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.flux[0]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[1]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[2]), NumpyArrayMatcher(geometric_fractions))])

        spacecraft_potential_initial_guess = swe_config['spacecraft_potential_initial_guess']
        halo_core_initial_guess = swe_config['core_halo_breakpoint_initial_guess']
        mock_find_breakpoints.assert_has_calls([
            call(swe_l2_data.energy, mock_average_flux.return_value, spacecraft_potential_initial_guess,
                 halo_core_initial_guess, 15, 90, swe_config),
            call(swe_l2_data.energy, mock_average_flux.return_value, spacecraft_potential_initial_guess,
                 halo_core_initial_guess, 12, 96, swe_config),
            call(swe_l2_data.energy, mock_average_flux.return_value, 14, 92, 16, 86, swe_config),
        ])

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.flux_by_pitch_angle, rebinned_by_pitch[0::2])
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_by_pitch_angle, rebinned_by_pitch[1::2])
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, swe_l2_data.epoch_delta)
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum, integrated_spectrum)
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_inbound, expected_inbound_spectrum)
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_outbound, expected_outbound_spectrum)

        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        self.assertEqual(mock_correct_and_rebin.call_args_list, [
            call_with_array_matchers(swe_l2_data.flux[0], swe_l2_data.energy - 12, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0],
                                     closest_mag_data[0], closest_swapi_data[0], swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[0], swe_l2_data.energy - 12, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0],
                                     closest_mag_data[0], closest_swapi_data[0], swe_config),
            call_with_array_matchers(swe_l2_data.flux[1], swe_l2_data.energy - 16, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1],
                                     closest_mag_data[1], closest_swapi_data[1], swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[1], swe_l2_data.energy - 16, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1],
                                     closest_mag_data[1], closest_swapi_data[1], swe_config),
            call_with_array_matchers(swe_l2_data.flux[2], swe_l2_data.energy - 19, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2],
                                     closest_mag_data[2], closest_swapi_data[2], swe_config),
            call_with_array_matchers(swe_l2_data.phase_space_density[2], swe_l2_data.energy - 19, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2],
                                     closest_mag_data[2], closest_swapi_data[2], swe_config)
        ])
        # mock_correct_and_rebin.assert_has_calls()
        mock_integrate_distribution_to_get_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch[1], swe_config),
            call(rebinned_by_pitch[3], swe_config),
            call(rebinned_by_pitch[5], swe_config)
        ])
        mock_integrate_distribution_to_get_inbound_and_outbound_1d_spectrum.assert_has_calls([
            call(rebinned_by_pitch[1], swe_config),
            call(rebinned_by_pitch[3], swe_config),
            call(rebinned_by_pitch[5], swe_config)
        ])
