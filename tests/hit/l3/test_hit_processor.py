from dataclasses import fields
from typing import Type, TypeVar
from unittest import TestCase
from unittest.mock import sentinel, patch, call, Mock

import numpy as np

from imap_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_processing.hit.l3.hit_processor import HitProcessor
from imap_processing.hit.l3.models import HitL2Data
from imap_processing.hit.l3.sectored_products.models import HitPitchAngleDataProduct
from imap_processing.models import MagL1dData, InputMetadata
from imap_processing.processor import Processor


class TestHitProcessor(TestCase):
    def test_is_a_processor(self):
        self.assertIsInstance(
            HitProcessor([], Mock()),
            Processor
        )

    @patch('imap_processing.hit.l3.hit_processor.imap_data_access.upload')
    @patch('imap_processing.hit.l3.hit_processor.convert_bin_high_low_to_center_delta')
    @patch('imap_processing.hit.l3.hit_processor.save_data')
    @patch('imap_processing.hit.l3.hit_processor.HITL3SectoredDependencies.fetch_dependencies')
    @patch('imap_processing.hit.l3.hit_processor.calculate_unit_vector')
    @patch('imap_processing.hit.l3.hit_processor.get_hit_bin_polar_coordinates')
    @patch('imap_processing.hit.l3.hit_processor.get_sector_unit_vectors')
    @patch('imap_processing.hit.l3.hit_processor.calculate_pitch_angle')
    @patch('imap_processing.hit.l3.hit_processor.calculate_gyrophase')
    @patch('imap_processing.hit.l3.hit_processor.rebin_by_pitch_angle_and_gyrophase')
    def test_process_pitch_angle_product(self, mock_rebin_by_pitch_angle_and_gyrophase, mock_calculate_gyrophase,
                                         mock_calculate_pitch_angle,
                                         mock_get_sector_unit_vectors, mock_get_hit_bin_polar_coordinates,
                                         mock_calculate_unit_vector,
                                         mock_fetch_dependencies, mock_save_data,
                                         mock_convert_bin_high_low_to_center_delta,
                                         mock_imap_data_access_upload):
        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="pitch-angle"
        )

        epochs = np.array([123, 234])
        epoch_deltas = np.array([12, 13])
        averaged_mag_vectors = [sentinel.mag_vector1, sentinel.mag_vector2]

        mock_convert_bin_high_low_to_center_delta.side_effect = [(sentinel.h_energy, sentinel.h_energy_delta),
                                                                 (sentinel.he4_energy, sentinel.he4_energy_delta),
                                                                 (sentinel.cno_energy, sentinel.cno_energy_delta),
                                                                 (sentinel.nemgsi_energy, sentinel.nemgsi_energy_delta),
                                                                 (sentinel.fe_energy, sentinel.fe_energy_delta)]

        mock_dependencies = Mock(spec=HITL3SectoredDependencies)
        mock_mag_data = self.create_dataclass_mock(MagL1dData)
        mock_mag_data.rebin_to = Mock()
        mock_mag_data.rebin_to.return_value = averaged_mag_vectors
        mock_dependencies.mag_l1d_data = mock_mag_data
        mock_hit_data = self.create_dataclass_mock(HitL2Data)
        mock_hit_data.epoch = epochs
        mock_hit_data.epoch_delta = epoch_deltas

        mock_hit_data.CNO = [sentinel.CNO_time1, sentinel.CNO_time2]
        mock_hit_data.helium4 = [sentinel.helium4_time1, sentinel.helium4_time2]
        mock_hit_data.hydrogen = [sentinel.hydrogen_time1, sentinel.hydrogen_time2]
        mock_hit_data.iron = [sentinel.iron_time1, sentinel.iron_time2]
        mock_hit_data.NeMgSi = [sentinel.NeMgSi_time1, sentinel.NeMgSi_time2]

        mock_dependencies.data = mock_hit_data
        mock_fetch_dependencies.return_value = mock_dependencies

        mock_calculate_unit_vector.side_effect = [sentinel.mag_unit_vector1, sentinel.mag_unit_vector2]
        mock_get_hit_bin_polar_coordinates.return_value = (
            sentinel.dec,  sentinel.inc, sentinel.dec_delta, sentinel.inc_delta)

        sector_unit_vectors = np.array([[1, 0, 0], [0, 1, 0]])
        mock_get_sector_unit_vectors.return_value = sector_unit_vectors

        pitch_angle1 = np.array([100])
        pitch_angle2 = np.array([200])

        gyrophase1 = np.array([1000])
        gyrophase2 = np.array([2000])

        mock_calculate_pitch_angle.side_effect = [pitch_angle1, pitch_angle2]
        mock_calculate_gyrophase.side_effect = [gyrophase1, gyrophase2]

        rebinned_pa_gyro_CNO_time1 = np.array([1])
        rebinned_pa_gyro_helium4_time1 = np.array([3])
        rebinned_pa_gyro_hydrogen_time1 = np.array([5])
        rebinned_pa_gyro_iron_time1 = np.array([7])
        rebinned_pa_gyro_NeMgSi_time1 = np.array([9])
        rebinned_pa_gyro_CNO_time2 = np.array([2])
        rebinned_pa_gyro_helium4_time2 = np.array([4])
        rebinned_pa_gyro_hydrogen_time2 = np.array([6])
        rebinned_pa_gyro_iron_time2 = np.array([8])
        rebinned_pa_gyro_NeMgSi_time2 = np.array([9])

        rebinned_pa_CNO_time1 = np.array([11])
        rebinned_pa_helium4_time1 = np.array([13])
        rebinned_pa_hydrogen_time1 = np.array([15])
        rebinned_pa_iron_time1 = np.array([17])
        rebinned_pa_NeMgSi_time1 = np.array([19])
        rebinned_pa_CNO_time2 = np.array([12])
        rebinned_pa_helium4_time2 = np.array([14])
        rebinned_pa_hydrogen_time2 = np.array([16])
        rebinned_pa_iron_time2 = np.array([18])
        rebinned_pa_NeMgSi_time2 = np.array([19])

        mock_rebin_by_pitch_angle_and_gyrophase.side_effect = [
            (rebinned_pa_gyro_CNO_time1,rebinned_pa_CNO_time1),
            (rebinned_pa_gyro_helium4_time1,rebinned_pa_helium4_time1),
            (rebinned_pa_gyro_hydrogen_time1,rebinned_pa_hydrogen_time1),
            (rebinned_pa_gyro_iron_time1,rebinned_pa_iron_time1),
            (rebinned_pa_gyro_NeMgSi_time1,rebinned_pa_NeMgSi_time1),
            (rebinned_pa_gyro_CNO_time2,rebinned_pa_CNO_time2),
            (rebinned_pa_gyro_helium4_time2,rebinned_pa_helium4_time2),
            (rebinned_pa_gyro_hydrogen_time2,rebinned_pa_hydrogen_time2),
            (rebinned_pa_gyro_iron_time2,rebinned_pa_iron_time2),
            (rebinned_pa_gyro_NeMgSi_time2,rebinned_pa_NeMgSi_time2)
        ]

        processor = HitProcessor(sentinel.upstream_dependency, input_metadata)
        processor.process()
        mock_fetch_dependencies.assert_called_once_with(sentinel.upstream_dependency)

        mock_mag_data.rebin_to.assert_called_once_with(mock_hit_data.epoch, mock_hit_data.epoch_delta)

        mock_calculate_unit_vector.assert_has_calls([
            call(sentinel.mag_vector1),
            call(sentinel.mag_vector2)
        ])
        mock_get_sector_unit_vectors.assert_called_once_with(sentinel.dec, sentinel.inc)

        mock_convert_bin_high_low_to_center_delta.assert_has_calls([
            call(mock_hit_data.h_energy_high, mock_hit_data.h_energy_low),
            call(mock_hit_data.he4_energy_high, mock_hit_data.he4_energy_low),
            call(mock_hit_data.cno_energy_high, mock_hit_data.cno_energy_low),
            call(mock_hit_data.nemgsi_energy_high, mock_hit_data.nemgsi_energy_low),
            call(mock_hit_data.fe_energy_high, mock_hit_data.fe_energy_low),
        ])

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
            call(sentinel.CNO_time1, pitch_angle1, gyrophase1, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.helium4_time1, pitch_angle1, gyrophase1, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.hydrogen_time1, pitch_angle1, gyrophase1, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.iron_time1, pitch_angle1, gyrophase1, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.NeMgSi_time1, pitch_angle1, gyrophase1, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),

            call(sentinel.CNO_time2, pitch_angle2, gyrophase2, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.helium4_time2, pitch_angle2, gyrophase2, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.hydrogen_time2, pitch_angle2, gyrophase2, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.iron_time2, pitch_angle2, gyrophase2, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
            call(sentinel.NeMgSi_time2, pitch_angle2, gyrophase2, 
                 number_of_pitch_angle_bins,
                 number_of_gyrophase_bins),
        ])

        saved_data_product: HitPitchAngleDataProduct = mock_save_data.call_args_list[0].args[0]

        np.testing.assert_array_equal(saved_data_product.epochs, epochs)
        np.testing.assert_array_equal(saved_data_product.epoch_deltas, epoch_deltas)

        np.testing.assert_array_equal(saved_data_product.pitch_angles, sentinel.dec)
        np.testing.assert_array_equal(saved_data_product.pitch_angle_deltas, sentinel.dec_delta)
        np.testing.assert_array_equal(saved_data_product.gyrophases, sentinel.inc)
        np.testing.assert_array_equal(saved_data_product.gyrophase_deltas, sentinel.inc_delta)

        np.testing.assert_array_equal(saved_data_product.h_fluxes,
                                      np.concatenate((rebinned_pa_gyro_hydrogen_time1, rebinned_pa_gyro_hydrogen_time2)))
        np.testing.assert_array_equal(saved_data_product.he4_fluxes,
                                      np.concatenate((rebinned_pa_gyro_helium4_time1, rebinned_pa_gyro_helium4_time2)))

        np.testing.assert_array_equal(saved_data_product.cno_fluxes,
                                      np.concatenate((rebinned_pa_gyro_CNO_time1, rebinned_pa_gyro_CNO_time2)))
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_fluxes,
                                      np.concatenate((rebinned_pa_gyro_NeMgSi_time1, rebinned_pa_gyro_NeMgSi_time2)))
        np.testing.assert_array_equal(saved_data_product.iron_fluxes,
                                      np.concatenate((rebinned_pa_gyro_iron_time1, rebinned_pa_gyro_iron_time2)))

        np.testing.assert_array_equal(saved_data_product.h_pa_fluxes,
                                      np.concatenate((rebinned_pa_hydrogen_time1, rebinned_pa_hydrogen_time2)))
        np.testing.assert_array_equal(saved_data_product.he4_pa_fluxes,
                                      np.concatenate((rebinned_pa_helium4_time1, rebinned_pa_helium4_time2)))

        np.testing.assert_array_equal(saved_data_product.cno_pa_fluxes,
                                      np.concatenate((rebinned_pa_CNO_time1, rebinned_pa_CNO_time2)))
        np.testing.assert_array_equal(saved_data_product.ne_mg_si_pa_fluxes,
                                      np.concatenate((rebinned_pa_NeMgSi_time1, rebinned_pa_NeMgSi_time2)))
        np.testing.assert_array_equal(saved_data_product.iron_pa_fluxes,
                                      np.concatenate((rebinned_pa_iron_time1, rebinned_pa_iron_time2)))

        self.assertIs(sentinel.h_energy, saved_data_product.h_energies)
        self.assertIs(sentinel.h_energy_delta, saved_data_product.h_energy_deltas)
        self.assertIs(sentinel.he4_energy, saved_data_product.he4_energies)
        self.assertIs(sentinel.he4_energy_delta, saved_data_product.he4_energy_deltas)
        self.assertIs(sentinel.cno_energy, saved_data_product.cno_energies)
        self.assertIs(sentinel.cno_energy_delta, saved_data_product.cno_energy_deltas)
        self.assertIs(sentinel.nemgsi_energy, saved_data_product.ne_mg_si_energies)
        self.assertIs(sentinel.nemgsi_energy_delta, saved_data_product.ne_mg_si_energy_deltas)
        self.assertIs(sentinel.fe_energy, saved_data_product.iron_energies)
        self.assertIs(sentinel.fe_energy_delta, saved_data_product.iron_energy_deltas)

        mock_imap_data_access_upload.assert_called_once_with(mock_save_data.return_value)

    @patch("imap_processing.hit.l3.hit_processor.imap_data_access.upload")
    @patch("imap_processing.hit.l3.hit_processor.save_data")
    @patch("imap_processing.hit.l3.hit_processor.process_pha_event")
    @patch("imap_processing.hit.l3.hit_processor.HitL3PhaDependencies.fetch_dependencies")
    @patch("imap_processing.hit.l3.hit_processor.PHAEventReader.read_all_pha_events")
    def test_process_direct_event_product(self, mock_read_all_events, mock_fetch_dependencies, mock_process_pha_event,
                                          mock_save_data, mock_imap_data_access_upload):
        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="direct-event"
        )

        dependencies = []
        mock_hit_l3_pha_dependencies = mock_fetch_dependencies.return_value
        mock_hit_l3_pha_dependencies.hit_l1_data.event_binary = [sentinel.binary_stream_1, sentinel.binary_stream_2]
        events_at_epoch_1 = [sentinel.raw_event_1, sentinel.raw_event_2]
        events_at_epoch_2 = [sentinel.raw_event_3, sentinel.raw_event_4]
        processed_pha_events = [sentinel.event_output_1, sentinel.event_output_2, sentinel.event_output_3,
                                sentinel.event_output_4]
        mock_process_pha_event.side_effect = processed_pha_events

        mock_read_all_events.side_effect = [events_at_epoch_1, events_at_epoch_2]

        processor = HitProcessor(dependencies, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(dependencies)
        mock_read_all_events.assert_has_calls([call(sentinel.binary_stream_1), call(sentinel.binary_stream_2)])

        mock_process_pha_event.assert_has_calls(
            [call(sentinel.raw_event_1, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup),
             call(sentinel.raw_event_2, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup),
             call(sentinel.raw_event_3, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup),
             call(sentinel.raw_event_4, mock_hit_l3_pha_dependencies.cosine_correction_lookup,
                  mock_hit_l3_pha_dependencies.gain_lookup, mock_hit_l3_pha_dependencies.range_fit_lookup)],
            any_order=False)

        mock_save_data.assert_called_once()
        direct_event_product = mock_save_data.call_args_list[0].args[0]
        self.assertEqual(processed_pha_events, direct_event_product.event_outputs)
        self.assertEqual(input_metadata, direct_event_product.input_metadata)

        mock_imap_data_access_upload.assert_called_once_with(mock_save_data.return_value)

    def test_raise_error_if_descriptor_doesnt_match(self):
        input_metadata = InputMetadata(
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="",
            descriptor="spectral-index"
        )
        processor = HitProcessor(Mock(), input_metadata)

        with self.assertRaises(ValueError) as e:
            processor.process()

        self.assertEqual(e.exception.args[0],
                         f"Don't know how to generate 'spectral-index' /n Known HIT l3 data products: 'pitch-angle', 'direct-event'.")

    T = TypeVar("T")

    def create_dataclass_mock(self, obj: Type[T]) -> T:
        return Mock(spec=[field.name for field in fields(obj)])
