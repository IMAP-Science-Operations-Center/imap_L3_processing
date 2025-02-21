import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, call

import numpy as np

from imap_processing.models import MagL1dData, InputMetadata, UpstreamDataDependency
from imap_processing.swe.l3.models import SweL2Data, SweConfiguration, SwapiL3aProtonData
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_processing.swe.swe_processor import SweProcessor
from tests.test_helpers import NumpyArrayMatcher


class TestSweProcessor(unittest.TestCase):
    @patch('imap_processing.swe.swe_processor.average_flux')
    @patch('imap_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_processing.swe.swe_processor.calculate_solar_wind_velocity_vector')
    @patch('imap_processing.swe.swe_processor.correct_and_rebin')
    def test_calculate_pitch_angle_products(self, mock_correct_and_rebin, mock_calculate_solar_wind_velocity_vector,
                                            mock_find_breakpoints,
                                            mock_average_flux):
        epochs = datetime.now() + np.arange(5) * timedelta(minutes=1)
        mag_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() - timedelta(seconds=15) + np.arange(10) * timedelta(minutes=.5)
        mock_find_breakpoints.return_value = (10, 80)
        pitch_angle_bins = [0, 90, 180]

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.repeat(timedelta(seconds=30), 5),
            phase_space_density=np.array([]),
            flux=np.arange(15).reshape(5, 3),
            energy=np.array([2, 4, 6]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.arange(10, 25).reshape(5, 3)
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
        solar_wind_velocity_vector = np.arange(8, 23).reshape(5, 3).repeat(2, 0)
        mock_calculate_solar_wind_velocity_vector.return_value = solar_wind_velocity_vector
        rebinned_by_pitch = [
            i + np.arange(len(swe_l2_data.energy) * len(pitch_angle_bins)).reshape(len(swe_l2_data.energy),
                                                                                   len(pitch_angle_bins)) for i in
            range(len(epochs))]
        mock_correct_and_rebin.side_effect = rebinned_by_pitch

        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        swe_config = SweConfiguration(
            geometric_fractions=geometric_fractions,
            pitch_angle_bins=pitch_angle_bins,
            pitch_angle_delta=[45, 45, 45],
            energy_bins=[1, 10, 100],
            energy_delta_plus=[2, 20, 200],
            energy_delta_minus=[8, 80, 800],
        )
        input_metadata = InputMetadata("swe", "l3", datetime(2025, 2, 21),
                                       datetime(2025, 2, 22), "v001")
        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)
        swe_processor = SweProcessor(dependencies=[], input_metadata=input_metadata)
        swe_l3_data = swe_processor.calculate_pitch_angle_products(swel3_dependency)

        self.assertEqual(5, mock_average_flux.call_count)
        self.assertEqual(5, mock_find_breakpoints.call_count)
        self.assertEqual(5, mock_correct_and_rebin.call_count)
        mock_calculate_solar_wind_velocity_vector.assert_called_once_with(
            swel3_dependency.swapi_l3a_proton_data.proton_sw_speed,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_clock_angle,
            swel3_dependency.swapi_l3a_proton_data.proton_sw_deflection_angle)

        mock_average_flux.assert_has_calls([
            call(NumpyArrayMatcher(swe_l2_data.flux[0]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[1]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[2]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[3]), NumpyArrayMatcher(geometric_fractions)),
            call(NumpyArrayMatcher(swe_l2_data.flux[4]), NumpyArrayMatcher(geometric_fractions))])

        mock_find_breakpoints.assert_has_calls([
            call(swe_l2_data.energy, mock_average_flux.return_value),
            call(swe_l2_data.energy, mock_average_flux.return_value),
            call(swe_l2_data.energy, mock_average_flux.return_value),
            call(swe_l2_data.energy, mock_average_flux.return_value),
            call(swe_l2_data.energy, mock_average_flux.return_value),
        ])

        self.assertEqual(UpstreamDataDependency("swe", "l3", datetime(2025, 2, 21),
                                                datetime(2025, 2, 22), "v001", "sci"), swe_l3_data.input_metadata)
        self.assertEqual(swe_l3_data.pitch_angle, swel3_dependency.configuration["pitch_angle_bins"])
        self.assertEqual(swe_l3_data.pitch_angle_delta, swel3_dependency.configuration["pitch_angle_delta"])
        self.assertEqual(swe_l3_data.energy, swel3_dependency.configuration["energy_bins"])
        self.assertEqual(swe_l3_data.energy_delta_plus, swel3_dependency.configuration["energy_delta_plus"])
        self.assertEqual(swe_l3_data.energy_delta_minus, swel3_dependency.configuration["energy_delta_minus"])
        np.testing.assert_array_equal(swe_l3_data.flux_by_pitch_angle, rebinned_by_pitch)
        np.testing.assert_array_equal(swe_l3_data.epoch_delta, swe_l2_data.epoch_delta)
        np.testing.assert_array_equal(swe_l3_data.epoch, swe_l2_data.epoch)

        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        mock_correct_and_rebin.assert_has_calls([
            call_with_array_matchers(swe_l2_data.flux[0], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0],
                                     mag_l1d_data.mag_data[0], solar_wind_velocity_vector[0], swe_config),
            call_with_array_matchers(swe_l2_data.flux[1], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1],
                                     mag_l1d_data.mag_data[2], solar_wind_velocity_vector[2], swe_config),
            call_with_array_matchers(swe_l2_data.flux[2], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2],
                                     mag_l1d_data.mag_data[4], solar_wind_velocity_vector[4], swe_config),
            call_with_array_matchers(swe_l2_data.flux[3], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[3],
                                     mag_l1d_data.mag_data[6], solar_wind_velocity_vector[6], swe_config),
            call_with_array_matchers(swe_l2_data.flux[4], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[4],
                                     mag_l1d_data.mag_data[8], solar_wind_velocity_vector[8], swe_config)
        ])
