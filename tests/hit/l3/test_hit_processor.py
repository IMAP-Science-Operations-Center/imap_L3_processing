from dataclasses import fields
from unittest import TestCase
from unittest.mock import sentinel, patch, call, Mock

import numpy as np

from imap_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_processing.hit.l3.hit_processor import HitProcessor
from imap_processing.hit.l3.models import HitL2Data
from imap_processing.models import MagL1dData
from imap_processing.processor import Processor


class TestHitProcessor(TestCase):
    def test_is_a_processor(self):
        self.assertIsInstance(
            HitProcessor([], Mock()),
            Processor
        )

    @patch('imap_processing.hit.l3.hit_processor.HITL3SectoredDependencies.fetch_dependencies')
    @patch('imap_processing.hit.l3.hit_processor.calculate_unit_vector')
    @patch('imap_processing.hit.l3.hit_processor.get_hit_bin_polar_coordinates')
    @patch('imap_processing.hit.l3.hit_processor.get_sector_unit_vectors')
    @patch('imap_processing.hit.l3.hit_processor.calculate_sector_areas')
    @patch('imap_processing.hit.l3.hit_processor.calculate_pitch_angle')
    @patch('imap_processing.hit.l3.hit_processor.calculate_gyrophase')
    @patch('imap_processing.hit.l3.hit_processor.rebin_by_pitch_angle_and_gyrophase')
    def test_can_fetch_dependencies(self, mock_rebin_by_pitch_angle_and_gyrophase, mock_calculate_gyrophase,
                                    mock_calculate_pitch_angle,
                                    mock_calculate_sector_areas,
                                    mock_get_sector_unit_vectors, mock_get_hit_bin_polar_coordinates,
                                    mock_calculate_unit_vector,
                                    mock_fetch_dependencies):
        epochs = [sentinel.time1, sentinel.time2]
        averaged_mag_vectors = [sentinel.mag_vector1, sentinel.mag_vector2]

        mock_dependencies = Mock(spec=HITL3SectoredDependencies)
        mock_mag_data = self.create_dataclass_mock(MagL1dData)
        mock_mag_data.rebin_to = Mock()
        mock_mag_data.rebin_to.return_value = averaged_mag_vectors
        mock_dependencies.mag_l1d_data = mock_mag_data
        mock_hit_data = self.create_dataclass_mock(HitL2Data)
        mock_hit_data.epoch = epochs

        mock_hit_data.CNO = [[sentinel.time1_energy], [sentinel.time2_energy]]
        mock_hit_data.helium4 = [[], []]
        mock_hit_data.hydrogen = [[], []]
        mock_hit_data.iron = [[], []]
        mock_hit_data.NeMgSi = [[], []]

        mock_dependencies.data = mock_hit_data
        mock_fetch_dependencies.return_value = mock_dependencies

        mock_calculate_unit_vector.side_effect = [sentinel.mag_unit_vector1, sentinel.mag_unit_vector2]
        mock_get_hit_bin_polar_coordinates.return_value = (
            sentinel.dec, sentinel.dec_delta, sentinel.inc, sentinel.inc_delta)

        sector_unit_vectors = np.array([[1, 0, 0], [0, 1, 0]])
        mock_get_sector_unit_vectors.return_value = sector_unit_vectors
        mock_calculate_sector_areas.return_value = sentinel.sector_areas

        mock_calculate_pitch_angle.side_effect = [sentinel.pitch_angle1, sentinel.pitch_angle2]
        mock_calculate_gyrophase.side_effect = [sentinel.gyrophase1, sentinel.gyrophase2]

        processor = HitProcessor(sentinel.upstream_dependency, Mock())
        processor.process()
        mock_fetch_dependencies.assert_called_once_with(sentinel.upstream_dependency)

        mock_mag_data.rebin_to.assert_called_once_with(mock_hit_data.epoch, mock_hit_data.epoch_delta)

        mock_calculate_unit_vector.assert_has_calls([
            call(sentinel.mag_vector1),
            call(sentinel.mag_vector2)
        ])
        mock_get_sector_unit_vectors.assert_called_once_with(sentinel.dec, sentinel.inc)
        mock_calculate_sector_areas.assert_called_once_with(sentinel.dec, sentinel.dec_delta, sentinel.inc_delta)

        self.assertEqual(2, mock_calculate_pitch_angle.call_count)
        np.testing.assert_array_equal(mock_calculate_pitch_angle.call_args_list[0].args[0], -sector_unit_vectors)
        self.assertEqual(mock_calculate_pitch_angle.call_args_list[0].args[1], sentinel.mag_unit_vector1)

        np.testing.assert_array_equal(mock_calculate_pitch_angle.call_args_list[1].args[0], -sector_unit_vectors)
        self.assertEqual(mock_calculate_pitch_angle.call_args_list[1].args[1], sentinel.mag_unit_vector2)

        self.assertEqual(2, mock_calculate_gyrophase.call_count)
        np.testing.assert_array_equal(mock_calculate_gyrophase.call_args_list[0].args[0], -sector_unit_vectors)
        self.assertEqual(mock_calculate_gyrophase.call_args_list[0].args[1], sentinel.mag_unit_vector1)

        np.testing.assert_array_equal(mock_calculate_gyrophase.call_args_list[1].args[0], -sector_unit_vectors)
        self.assertEqual(mock_calculate_gyrophase.call_args_list[1].args[1], sentinel.mag_unit_vector2)

        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15

        mock_rebin_by_pitch_angle_and_gyrophase.assert_has_calls([
            call(sentinel.time1_energy, sentinel.pitch_angle1, sentinel.gyrophase1, sentinel.sector_areas,
                 number_of_pitch_angle_bins, number_of_gyrophase_bins),
            call(sentinel.time2_energy, sentinel.pitch_angle2, sentinel.gyrophase2, sentinel.sector_areas,
                 number_of_pitch_angle_bins, number_of_gyrophase_bins)
        ])

    def create_dataclass_mock(self, obj):
        return Mock(spec=[field.name for field in fields(obj)])
