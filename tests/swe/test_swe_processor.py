import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, call, Mock, sentinel

import numpy as np

from imap_l3_processing.models import MagL1dData, InputMetadata, UpstreamDataDependency
from imap_l3_processing.swe.l3.models import SweL2Data, SwapiL3aProtonData
from imap_l3_processing.swe.l3.science.moment_calculations import Moments
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.swe_processor import SweProcessor
from tests.test_helpers import NumpyArrayMatcher, build_swe_configuration, create_dataclass_mock


class TestSweProcessor(unittest.TestCase):
    @patch('imap_l3_processing.swe.swe_processor.upload')
    @patch('imap_l3_processing.swe.swe_processor.save_data')
    @patch('imap_l3_processing.swe.swe_processor.SweL3Dependencies.fetch_dependencies')
    @patch('imap_l3_processing.swe.swe_processor.SweProcessor.calculate_pitch_angle_products')
    def test_process(self, mock_calculate_pitch_angle_products, mock_fetch_dependencies, mock_save_data, mock_upload):
        mock_dependencies = Mock()
        mock_input_metadata = Mock()
        swe_processor = SweProcessor(mock_dependencies, mock_input_metadata)
        swe_processor.process()

        mock_fetch_dependencies.assert_called_once_with(mock_dependencies)
        mock_calculate_pitch_angle_products.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_once_with(mock_calculate_pitch_angle_products.return_value)
        mock_upload.assert_called_once_with(mock_save_data.return_value)

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

        self.assertEqual(3, mock_average_over_look_directions.call_count)
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
        mock_average_over_look_directions.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), NumpyArrayMatcher(geometric_fractions), 1e-32),
            call(NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), NumpyArrayMatcher(geometric_fractions), 1e-32),
            call(NumpyArrayMatcher(swe_l2_data.phase_space_density[2]), NumpyArrayMatcher(geometric_fractions), 1e-32)])

        mock_find_breakpoints.assert_has_calls([
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 15, 15],
                 [90, 90, 90], swe_config),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 15, 12],
                 [90, 90, 96], swe_config),
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 12, 16],
                 [90, 96, 86], swe_config),
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

        actual_calls = mock_correct_and_rebin.call_args_list

        expected_calls = [
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
        ]
        self.assertEqual(actual_calls, expected_calls)
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

    def test_calculate_pitch_angle_products_makes_nan_if_no_mag_close_enough(self):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_epochs = np.array([datetime(2025, 3, 6, 0, 1, 30)])
        swapi_epochs = np.array([datetime(2025, 3, 6)])

        pitch_angle_bins = [70, 100, 130]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), num_epochs),
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 20,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
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
        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_pitch_angle_products(swel3_dependency)

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.flux_by_pitch_angle,
                                      np.full((len(epochs), len(energy_bins), len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.phase_space_density_by_pitch_angle,
                                      np.full((len(epochs), len(energy_bins), len(pitch_angle_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, swe_l2_data.epoch_delta)
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum, np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_inbound,
                                      np.full((len(epochs), len(energy_bins)), np.nan))
        np.testing.assert_array_equal(swe_l3_data.energy_spectrum_outbound,
                                      np.full((len(epochs), len(energy_bins)), np.nan))

    def test_calculate_pitch_angle_products_without_mocks(self):
        epochs = np.array([datetime(2025, 3, 6)])
        mag_epochs = np.array([datetime(2025, 3, 6, 0, 0, 30)])
        swapi_epochs = np.array([datetime(2025, 3, 6)])

        pitch_angle_bins = [70, 100, 130]

        num_energies = 9
        num_epochs = 1
        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), num_epochs),
            phase_space_density=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5,
                                                                                     7) + 100,
            flux=np.arange(num_epochs * num_energies * 5 * 7).reshape(num_epochs, num_energies, 5, 7),
            energy=np.arange(num_energies) + 20,
            inst_el=np.array([-30, -20, -10, 0, 10, 20, 30]),
            inst_az_spin_sector=np.arange(num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
            acquisition_time=np.linspace(datetime(2025, 3, 6), datetime(2025, 3, 6, 0, 1),
                                         num_epochs * num_energies * 5).reshape(num_epochs, num_energies, 5),
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
        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_pitch_angle_products(swel3_dependency)

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_allclose(swe_l3_data.flux_by_pitch_angle,
                                   np.array([[[np.nan, 73.003799, 180.513816],
                                              [111.982682, 159.352476, 281.418325],
                                              [209.853169, 314.550222, np.nan]]]))
        np.testing.assert_allclose(swe_l3_data.phase_space_density_by_pitch_angle,
                                   np.array([[[np.nan, 194.772034, 270.312835],
                                              [211.672665, 273.136802, 363.195552],
                                              [313.860051, 400.23009, np.nan]]]))
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, swe_l2_data.epoch_delta)
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)
        np.testing.assert_allclose(swe_l3_data.energy_spectrum, np.array([[104.427758, 195.333344, 180.401159]]))
        np.testing.assert_allclose(swe_l3_data.energy_spectrum_inbound, np.array([[0, 104.147588, 154.42602]]))
        np.testing.assert_allclose(swe_l3_data.energy_spectrum_outbound,
                                   np.array([[208.855516, 286.519101, 206.376298]]))

    @patch('imap_l3_processing.swe.swe_processor.halotrunc')
    @patch('imap_l3_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.filter_and_flatten_regress_parameters')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.regress')
    @patch('imap_l3_processing.swe.swe_processor.calculate_fit_temperature_density_velocity')
    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    @patch('imap_l3_processing.swe.swe_processor.spice_wrapper')
    def test_calculate_moment_products(self, mock_spice_wrapper, mock_rotate_temperature,
                                       mock_rotate_dps_vector_to_rtn,
                                       mock_calculate_fit_temperature_density_velocity, mock_regress,
                                       mock_calculate_velocity_in_dsp_frame_km_s,
                                       mock_compute_maxwellian_weight_factors,
                                       mock_filter_and_flatten_regress_parameters,
                                       mock_average_over_look_directions, mock_find_breakpoints, mock_halotrunc):
        epochs = datetime.now() + np.arange(2) * timedelta(minutes=1)
        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]

        swe_config = build_swe_configuration(
            geometric_fractions=geometric_fractions,
            spacecraft_potential_initial_guess=15,
            core_halo_breakpoint_initial_guess=90,
        )

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 2),
            phase_space_density=np.arange(9).reshape(3, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=np.array([9, 10, 12, 14, 36, 54, 96, 102, 112, 156]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
        )

        expected_breakpoint_1 = (12, 96)
        expected_breakpoint_2 = (14, 54)
        mock_find_breakpoints.side_effect = [
            expected_breakpoint_1,
            expected_breakpoint_2
        ]

        velocity_in_dsp_frame_km_s = [
            np.full(shape=(24, 30, 7, 3), fill_value=1),
            np.full(shape=(24, 30, 7, 3), fill_value=2)
        ]

        mock_calculate_velocity_in_dsp_frame_km_s.side_effect = velocity_in_dsp_frame_km_s

        maxwellian_weight_factors = [np.full(shape=(24, 30, 7), fill_value=1),
                                     np.full(shape=(24, 30, 7), fill_value=2)]

        mock_compute_maxwellian_weight_factors.side_effect = maxwellian_weight_factors

        filtered_velocity_vectors = [np.full(shape=(1080, 3), fill_value=1.1),
                                     np.full(shape=(1080, 3), fill_value=5.1),
                                     np.full(shape=(1080, 3), fill_value=2.1),
                                     np.full(shape=(1080, 3), fill_value=6.1)]
        filtered_weights = [np.full(shape=(1080,), fill_value=1.2),
                            np.full(shape=(1080,), fill_value=5.2),
                            np.full(shape=(1080,), fill_value=2.2),
                            np.full(shape=(1080,), fill_value=6.2)]

        filtered_yreg = [
            np.full(shape=(1080,), fill_value=1.3),
            np.full(shape=(1080,), fill_value=5.3),
            np.full(shape=(1080,), fill_value=2.3),
            np.full(shape=(1080,), fill_value=6.3),
        ]

        mock_filter_and_flatten_regress_parameters.side_effect = [
            (
                filtered_velocity_vectors[0],
                filtered_weights[0],
                filtered_yreg[0]
            ),
            (
                filtered_velocity_vectors[1],
                filtered_weights[1],
                filtered_yreg[1],
            ),
            (
                filtered_velocity_vectors[2],
                filtered_weights[2],
                filtered_yreg[2]
            ),
            (
                filtered_velocity_vectors[3],
                filtered_weights[3],
                filtered_yreg[3],
            ),
        ]

        core_moments1 = create_dataclass_mock(Moments, density=10, velocity_x=5, velocity_y=6, velocity_z=7)
        halo_moments1 = create_dataclass_mock(Moments, density=sentinel.halo_density_1, velocity_x=5, velocity_y=6,
                                              velocity_z=7)
        core_moments2 = create_dataclass_mock(Moments, density=10, velocity_x=8, velocity_y=9, velocity_z=10)
        halo_moments2 = create_dataclass_mock(Moments, density=sentinel.halo_density_2, velocity_x=8, velocity_y=9,
                                              velocity_z=10)

        mock_halotrunc.side_effect = [
            10,
            10
        ]

        mock_calculate_fit_temperature_density_velocity.side_effect = [core_moments1, halo_moments1, core_moments2,
                                                                       halo_moments2]

        mock_regress.side_effect = [(sentinel.first_core_regress_return, 0),
                                    (sentinel.first_halo_moment_regress_return, 0),
                                    (sentinel.second_core_regress_return, 0),
                                    (sentinel.second_halo_moment_regress_return, 0)]

        mock_average_over_look_directions.return_value = np.array([5, 10, 15])
        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swel3_dependency = SweL3Dependencies(swe_l2_data, Mock(), Mock(), swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)

        swe_processor.calculate_moment_products(swel3_dependency)

        mock_spice_wrapper.furnish.assert_called_once()

        self.assertEqual(2, mock_average_over_look_directions.call_count)
        mock_average_over_look_directions.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), NumpyArrayMatcher(geometric_fractions), 1e-32),
            call(NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), NumpyArrayMatcher(geometric_fractions), 1e-32)])

        mock_find_breakpoints.assert_has_calls([
            call(swe_l2_data.energy, mock_average_over_look_directions.return_value,
                 [15, 15, 15],
                 [90, 90, 90], swe_config),

            call(swe_l2_data.energy, mock_average_over_look_directions.return_value, [15, 15, 12],
                 [90, 90, 96], swe_config)
        ])

        calculate_velocity_call_1 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[0]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_1[0], calculate_velocity_call_1.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[0], calculate_velocity_call_1.args[2])

        calculate_velocity_call_2 = mock_calculate_velocity_in_dsp_frame_km_s.mock_calls[1]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_2[0], calculate_velocity_call_2.args[0])
        np.testing.assert_array_equal(swe_l2_data.inst_el, calculate_velocity_call_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.inst_az_spin_sector[1], calculate_velocity_call_2.args[2])

        np.testing.assert_array_equal(np.reshape(np.arange(24 * 30 * 7), (24, 30, 7)) * 1000,
                                      mock_compute_maxwellian_weight_factors.mock_calls[0].args[0])
        np.testing.assert_array_equal(np.reshape(np.arange(24 * 30 * 7), (24, 30, 7)) * 1000,
                                      mock_compute_maxwellian_weight_factors.mock_calls[1].args[0])

        core_filter_and_flatten_call_1 = mock_filter_and_flatten_regress_parameters.mock_calls[0]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_1[0],
                                      core_filter_and_flatten_call_1.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0],
                                      core_filter_and_flatten_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], core_filter_and_flatten_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      core_filter_and_flatten_call_1.args[3])
        self.assertEqual(2, core_filter_and_flatten_call_1.args[4])
        self.assertEqual(6, core_filter_and_flatten_call_1.args[5])

        halo_filter_and_flatten_call_1 = mock_filter_and_flatten_regress_parameters.mock_calls[1]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_1[0],
                                      halo_filter_and_flatten_call_1.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[0],
                                      halo_filter_and_flatten_call_1.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[0], halo_filter_and_flatten_call_1.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[0],
                                      halo_filter_and_flatten_call_1.args[3])
        self.assertEqual(6, halo_filter_and_flatten_call_1.args[4])
        self.assertEqual(10, halo_filter_and_flatten_call_1.args[5])

        core_filter_and_flatten_call_2 = mock_filter_and_flatten_regress_parameters.mock_calls[2]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_2[0],
                                      core_filter_and_flatten_call_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1],
                                      core_filter_and_flatten_call_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], core_filter_and_flatten_call_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      core_filter_and_flatten_call_2.args[3])
        self.assertEqual(3, core_filter_and_flatten_call_2.args[4])
        self.assertEqual(5, core_filter_and_flatten_call_2.args[5])

        halo_filter_and_flatten_call_2 = mock_filter_and_flatten_regress_parameters.mock_calls[3]
        np.testing.assert_array_equal(swe_l2_data.energy - expected_breakpoint_2[0],
                                      halo_filter_and_flatten_call_2.args[0])
        np.testing.assert_array_equal(velocity_in_dsp_frame_km_s[1],
                                      halo_filter_and_flatten_call_2.args[1])
        np.testing.assert_array_equal(swe_l2_data.phase_space_density[1], halo_filter_and_flatten_call_2.args[2])
        np.testing.assert_array_equal(maxwellian_weight_factors[1],
                                      halo_filter_and_flatten_call_2.args[3])
        self.assertEqual(5, halo_filter_and_flatten_call_2.args[4])
        self.assertEqual(10, halo_filter_and_flatten_call_2.args[5])

        core_regress_call_1 = mock_regress.mock_calls[0]
        np.testing.assert_array_equal(filtered_velocity_vectors[0], core_regress_call_1.args[0])
        np.testing.assert_array_equal(filtered_weights[0], core_regress_call_1.args[1])
        np.testing.assert_array_equal(filtered_yreg[0], core_regress_call_1.args[2])

        halo_regress_call_1 = mock_regress.mock_calls[1]
        np.testing.assert_array_equal(filtered_velocity_vectors[1], halo_regress_call_1.args[0])
        np.testing.assert_array_equal(filtered_weights[1], halo_regress_call_1.args[1])
        np.testing.assert_array_equal(filtered_yreg[1], halo_regress_call_1.args[2])

        core_regress_call_2 = mock_regress.mock_calls[2]
        np.testing.assert_array_equal(filtered_velocity_vectors[2], core_regress_call_2.args[0])
        np.testing.assert_array_equal(filtered_weights[2], core_regress_call_2.args[1])
        np.testing.assert_array_equal(filtered_yreg[2], core_regress_call_2.args[2])

        halo_regress_call_2 = mock_regress.mock_calls[3]
        np.testing.assert_array_equal(filtered_velocity_vectors[3], halo_regress_call_2.args[0])
        np.testing.assert_array_equal(filtered_weights[3], halo_regress_call_2.args[1])
        np.testing.assert_array_equal(filtered_yreg[3], halo_regress_call_2.args[2])

        mock_calculate_fit_temperature_density_velocity.assert_has_calls(
            [call(sentinel.first_core_regress_return),
             call(sentinel.first_halo_moment_regress_return),
             call(sentinel.second_core_regress_return),
             call(sentinel.second_halo_moment_regress_return), ])

        mock_halotrunc.assert_has_calls([
            call(halo_moments1, expected_breakpoint_1[1], expected_breakpoint_1[0]),
            call(halo_moments2, expected_breakpoint_2[1], expected_breakpoint_2[0])
        ])

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

    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.halotrunc')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.regress')
    @patch('imap_l3_processing.swe.swe_processor.calculate_fit_temperature_density_velocity')
    @patch('imap_l3_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_l3_processing.swe.swe_processor.filter_and_flatten_regress_parameters')
    def test_moment_fit_should_be_retried_until_returned_density_is_greater_than_zero_and_less_than_rolling_average_density(
            self,
            mock_filter_and_flatten_regress_parameters,
            mock_find_breakpoints,
            mock_calculate_fit_temperature_density_velocity,
            mock_regress, mock_calculate_velocity,
            mock_compute_maxwellian_weights,
            mock_halotrunc,
            _, __, ___):
        epochs = datetime.now() + np.arange(2) * timedelta(minutes=1)
        swe_config = build_swe_configuration()

        energies = np.array([1.2, 3.4, 4.6, 5.9, 8.7, 9.1, 9.2, 10.5, 11.9, 12.8, 13.9, 14.6])

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 2),
            phase_space_density=np.arange(2 * 7 * 3).reshape(2, 7, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=energies,
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
        )

        core_breakpoint = 3.4
        core_halo_breakpoint = 9.2
        mock_find_breakpoints.return_value = (core_breakpoint, core_halo_breakpoint)

        expected_core_energy_start_index = 1
        expected_core_energy_end_index = 6

        expected_halo_energy_end_index = len(energies)

        # @formatter:off
        mock_filter_and_flatten_regress_parameters.side_effect = [
            (sentinel.time1_core_filtered_vecs_1, sentinel.time1_core_filtered_weights_1, sentinel.time1_core_filtered_yreg_1),
            (sentinel.time1_core_filtered_vecs_2, sentinel.time1_core_filtered_weights_2, sentinel.time1_core_filtered_yreg_2),
            (sentinel.time1_halo_filtered_vecs_1, sentinel.time1_halo_filtered_weights_1, sentinel.time1_halo_filtered_yreg_1),
            (sentinel.time1_halo_filtered_vecs_2, sentinel.time1_halo_filtered_weights_2, sentinel.time1_halo_filtered_yreg_2),

            (sentinel.time2_core_filtered_vecs_1, sentinel.time2_core_filtered_weights_1, sentinel.time2_core_filtered_yreg_1),
            (sentinel.time2_core_filtered_vecs_2, sentinel.time2_core_filtered_weights_2, sentinel.time2_core_filtered_yreg_2),
            (sentinel.time2_core_filtered_vecs_3, sentinel.time2_core_filtered_weights_3, sentinel.time2_core_filtered_yreg_3),
            (sentinel.time2_halo_filtered_vecs_1, sentinel.time2_halo_filtered_weights_1, sentinel.time2_halo_filtered_yreg_1),
            (sentinel.time2_halo_filtered_vecs_2, sentinel.time2_halo_filtered_weights_2, sentinel.time2_halo_filtered_yreg_2),
            (sentinel.time2_halo_filtered_vecs_3, sentinel.time2_halo_filtered_weights_3, sentinel.time2_halo_filtered_yreg_3),
        ]
        # @formatter:on

        mock_regress.side_effect = [(sentinel.time1_core_regress_1, 0),
                                    (sentinel.time1_core_regress_2, 0),
                                    (sentinel.time1_halo_regress_1, 0),
                                    (sentinel.time1_halo_regress_2, 0),

                                    (sentinel.time2_core_regress_1, 0),
                                    (sentinel.time2_core_regress_2, 0),
                                    (sentinel.time2_core_regress_3, 0),
                                    (sentinel.time2_halo_regress_1, 0),
                                    (sentinel.time2_halo_regress_2, 0),
                                    (sentinel.time2_halo_regress_3, 0)
                                    ]

        low_density = -1

        time1_core_valid_density = 10
        time2_core_valid_density = 13
        core_density_greater_than_config_value = 1.85 * 100 + 1
        core_density_greater_than_rolling_average = (100 + 100 + time1_core_valid_density) / 3 * 1.85 + 1

        core_momements_time1 = [(create_dataclass_mock(Moments, density=core_density_greater_than_config_value)),
                                (create_dataclass_mock(Moments, density=time1_core_valid_density)),
                                ]

        core_momements_time2 = [
            (create_dataclass_mock(Moments, density=core_density_greater_than_rolling_average)),
            (create_dataclass_mock(Moments, density=low_density)),
            (create_dataclass_mock(Moments, density=time2_core_valid_density))]

        time1_halo_valid_density = 8
        time2_halo_valid_density = 19
        halo_density_greater_than_config_value = 1.65 * 25 + 1
        halo_density_greater_than_rolling_average = (25 + 25 + time1_halo_valid_density) / 3 * 1.65 + 1

        halo_moments_time1 = [
            (create_dataclass_mock(Moments, density=sentinel.fit_density_1)),
            (create_dataclass_mock(Moments, density=sentinel.fit_density_2)),
        ]

        halo_moments_time2 = [
            (create_dataclass_mock(Moments, density=sentinel.fit_density_3)),
            (create_dataclass_mock(Moments, density=sentinel.fit_density_4)),
            (create_dataclass_mock(Moments, density=sentinel.fit_density_5))
        ]

        mock_halotrunc.side_effect = [
            halo_density_greater_than_config_value,
            time1_halo_valid_density,
            halo_density_greater_than_rolling_average,
            low_density,
            time2_halo_valid_density
        ]

        mock_calculate_fit_temperature_density_velocity.side_effect = core_momements_time1 + halo_moments_time1 + core_momements_time2 + halo_moments_time2

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swel3_dependency = SweL3Dependencies(swe_l2_data, Mock(), Mock(), swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)

        swe_processor.calculate_moment_products(swel3_dependency)

        self.assertEqual(10, mock_filter_and_flatten_regress_parameters.call_count)
        mock_filter_and_flatten_regress_parameters.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index - 1),

            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, expected_halo_energy_end_index),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, expected_halo_energy_end_index - 1),

            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index - 1),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index - 2),

            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, expected_halo_energy_end_index),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, expected_halo_energy_end_index - 1),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[1]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, expected_halo_energy_end_index - 2),

        ])

        # @formatter:off
        self.assertEqual(10, mock_regress.call_count)
        mock_regress.assert_has_calls([
            call(sentinel.time1_core_filtered_vecs_1, sentinel.time1_core_filtered_weights_1, sentinel.time1_core_filtered_yreg_1),
            call(sentinel.time1_core_filtered_vecs_2, sentinel.time1_core_filtered_weights_2, sentinel.time1_core_filtered_yreg_2),
            call(sentinel.time1_halo_filtered_vecs_1, sentinel.time1_halo_filtered_weights_1, sentinel.time1_halo_filtered_yreg_1),
            call(sentinel.time1_halo_filtered_vecs_2, sentinel.time1_halo_filtered_weights_2, sentinel.time1_halo_filtered_yreg_2),

            call(sentinel.time2_core_filtered_vecs_1, sentinel.time2_core_filtered_weights_1, sentinel.time2_core_filtered_yreg_1),
            call(sentinel.time2_core_filtered_vecs_2, sentinel.time2_core_filtered_weights_2, sentinel.time2_core_filtered_yreg_2),
            call(sentinel.time2_core_filtered_vecs_3, sentinel.time2_core_filtered_weights_3, sentinel.time2_core_filtered_yreg_3),
            call(sentinel.time2_halo_filtered_vecs_1, sentinel.time2_halo_filtered_weights_1, sentinel.time2_halo_filtered_yreg_1),
            call(sentinel.time2_halo_filtered_vecs_2, sentinel.time2_halo_filtered_weights_2, sentinel.time2_halo_filtered_yreg_2),
            call(sentinel.time2_halo_filtered_vecs_3, sentinel.time2_halo_filtered_weights_3, sentinel.time2_halo_filtered_yreg_3),
        ])
        # @formatter:on

        self.assertEqual(10, mock_calculate_fit_temperature_density_velocity.call_count)
        mock_calculate_fit_temperature_density_velocity.assert_has_calls([
            call(sentinel.time1_core_regress_1),
            call(sentinel.time1_core_regress_2),
            call(sentinel.time1_halo_regress_1),
            call(sentinel.time1_halo_regress_2),

            call(sentinel.time2_core_regress_1),
            call(sentinel.time2_core_regress_2),
            call(sentinel.time2_core_regress_3),
            call(sentinel.time2_halo_regress_1),
            call(sentinel.time2_halo_regress_2),
            call(sentinel.time2_halo_regress_3)
        ])

        self.assertEqual(5, mock_halotrunc.call_count)
        mock_halotrunc.assert_has_calls([
            call(halo_moments_time1[0], core_halo_breakpoint, core_breakpoint),
            call(halo_moments_time1[1], core_halo_breakpoint, core_breakpoint),
            call(halo_moments_time2[0], core_halo_breakpoint, core_breakpoint),
            call(halo_moments_time2[1], core_halo_breakpoint, core_breakpoint),
            call(halo_moments_time2[2], core_halo_breakpoint, core_breakpoint),
        ])

    @patch('imap_l3_processing.swe.swe_processor.rotate_dps_vector_to_rtn')
    @patch('imap_l3_processing.swe.swe_processor.rotate_temperature')
    @patch('imap_l3_processing.swe.swe_processor.average_over_look_directions')
    @patch('imap_l3_processing.swe.swe_processor.halotrunc')
    @patch('imap_l3_processing.swe.swe_processor.compute_maxwellian_weight_factors')
    @patch('imap_l3_processing.swe.swe_processor.calculate_velocity_in_dsp_frame_km_s')
    @patch('imap_l3_processing.swe.swe_processor.regress')
    @patch('imap_l3_processing.swe.swe_processor.calculate_fit_temperature_density_velocity')
    @patch('imap_l3_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_l3_processing.swe.swe_processor.filter_and_flatten_regress_parameters')
    def test_moment_fit_should_be_ran_while_number_of_energies_between_halo_and_core_is_greater_than_3(self,
                                                                                                       mock_filter_and_flatten_regress_parameters,
                                                                                                       mock_find_breakpoints,
                                                                                                       mock_calculate_fit_temperature_density_velocity,
                                                                                                       mock_regress,
                                                                                                       mock_calculate_velocity,
                                                                                                       mock_compute_maxwellian_weights,
                                                                                                       mock_halotrunc,
                                                                                                       _, __, ___):
        epochs = datetime.now() + np.arange(1) * timedelta(minutes=1)
        swe_config = build_swe_configuration()

        energies = np.array([1.2, 3.4, 4.6, 5.9, 8.7, 9.2, 10.5])

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 1),
            phase_space_density=np.arange(7 * 3).reshape(1, 7, 3) + 100,
            flux=np.arange(9).reshape(3, 3),
            energy=energies,
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 19).reshape(3, 3),
            acquisition_time=np.array([]),
        )

        core_breakpoint = 3.4
        core_halo_breakpoint = 8.7
        mock_find_breakpoints.return_value = (core_breakpoint, core_halo_breakpoint)

        expected_core_energy_start_index = 1
        expected_core_energy_end_index = 4

        mock_filter_and_flatten_regress_parameters.side_effect = [
            (sentinel.core_filtered_vecs_1, sentinel.core_filtered_weights_1, sentinel.core_filtered_yreg_1),
            (sentinel.halo_filtered_vecs_1, sentinel.halo_filtered_weights_1, sentinel.halo_filtered_yreg_1)]

        mock_regress.side_effect = [(sentinel.core_regress_return, 0), (sentinel.halo_regress_return, 0)]

        core_moments = create_dataclass_mock(Moments, density=-1)
        halo_moments = create_dataclass_mock(Moments)

        mock_halotrunc.side_effect = [-1]

        mock_calculate_fit_temperature_density_velocity.side_effect = [core_moments,
                                                                       halo_moments]

        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")

        swel3_dependency = SweL3Dependencies(swe_l2_data, Mock(), Mock(), swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)

        swe_processor.calculate_moment_products(swel3_dependency)

        mock_filter_and_flatten_regress_parameters.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_start_index, expected_core_energy_end_index),
            call(NumpyArrayMatcher(swe_l2_data.energy - core_breakpoint), mock_calculate_velocity.return_value,
                 NumpyArrayMatcher(swe_l2_data.phase_space_density[0]), mock_compute_maxwellian_weights.return_value,
                 expected_core_energy_end_index, len(energies)),
        ])

        mock_regress.assert_has_calls([call(sentinel.core_filtered_vecs_1, sentinel.core_filtered_weights_1,
                                            sentinel.core_filtered_yreg_1),
                                       call(sentinel.halo_filtered_vecs_1, sentinel.halo_filtered_weights_1,
                                            sentinel.halo_filtered_yreg_1)])

        mock_calculate_fit_temperature_density_velocity.assert_has_calls(
            [call(sentinel.core_regress_return), call(sentinel.halo_regress_return)])
