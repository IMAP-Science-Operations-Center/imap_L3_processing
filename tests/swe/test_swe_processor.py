import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call

import numpy as np

from imap_processing.models import MagL1dData
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData
from imap_processing.swe.l3.models import SweL2Data
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies, SweConfiguration
from imap_processing.swe.swe_processor import SweProcessor
from tests.test_helpers import NumpyArrayMatcher


class TestSweProcessor(unittest.TestCase):
    @patch('imap_processing.swe.swe_processor.average_flux')
    @patch('imap_processing.swe.swe_processor.find_breakpoints')
    @patch('imap_processing.swe.swe_processor.calculate_solar_wind_velocity_vector')
    @patch('imap_processing.swe.swe_processor.correct_and_rebin')
    def test_something(self, mock_correct_and_rebin, mock_calculate_solar_wind_velocity_vector, mock_find_breakpoints,
                       mock_average_flux):
        epochs = datetime.now() + np.arange(5) * timedelta(minutes=1)
        mag_epochs = datetime.now() + np.arange(10) * timedelta(minutes=.5)
        swapi_epochs = datetime.now() + np.arange(10) * timedelta(minutes=1)
        mock_find_breakpoints.return_value = (10, 80)

        swe_l2_data = SweL2Data(
            epoch=epochs,
            epoch_delta=np.array(timedelta(seconds=30)) * 5,
            phase_space_density=np.array([]),
            flux=np.repeat([[1, 2, 3]], 5, axis=0),
            energy=np.array([2, 4, 6]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.repeat([[1, 2, 3]], 5, axis=0)
        )

        mag_l1d_data = MagL1dData(
            epoch=mag_epochs,
            mag_data=np.repeat([[1, 0, 0]], 10, axis=0)
        )

        swapi_l3a_proton_data = SwapiL3ProtonSolarWindData(
            epoch=swapi_epochs,
            proton_sw_speed=np.array([]),
            proton_sw_density=np.array([]),
            proton_sw_temperature=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
            input_metadata=np.array([])
        )
        mock_average_flux.return_value = np.array([5, 10, 15])
        mock_calculate_solar_wind_velocity_vector.return_value = np.full((10, 3), 10)

        geometric_fractions = [0.0697327, 0.138312, 0.175125, 0.181759,
                               0.204686, 0.151448, 0.0781351]
        swe_config = SweConfiguration(
            geometric_fractions=geometric_fractions
        )
        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_proton_data, swe_config)

        swe_processor = SweProcessor(dependencies=[], input_metadata=Mock())
        swel3_data = swe_processor.calculate_pitch_angle_products(swel3_dependency)

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

        def call_with_array_matchers(*args):
            return call(*[NumpyArrayMatcher(x) for x in args])

        mock_correct_and_rebin.assert_has_calls([
            call_with_array_matchers(swe_l2_data.flux[0], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[0],
                                     [1, 0, 0], [10, 10, 10], [0, 90, 180], [1, 10, 100]),
            call_with_array_matchers(swe_l2_data.flux[1], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[1],
                                     [1, 0, 0], [10, 10, 10], [0, 90, 180], [1, 10, 100]),
            call_with_array_matchers(swe_l2_data.flux[2], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[2],
                                     [1, 0, 0], [10, 10, 10], [0, 90, 180], [1, 10, 100]),
            call_with_array_matchers(swe_l2_data.flux[3], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[3],
                                     [1, 0, 0], [10, 10, 10], [0, 90, 180], [1, 10, 100]),
            call_with_array_matchers(swe_l2_data.flux[4], swe_l2_data.energy - 10, swe_l2_data.inst_el,
                                     swe_l2_data.inst_az_spin_sector[4],
                                     [1, 0, 0], [10, 10, 10], [0, 90, 180], [1, 10, 100])
        ])
