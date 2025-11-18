import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, sentinel, Mock, call, MagicMock

import numpy as np

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.direct_event.science.tof_lookup import TOFLookup, EnergyPerNuc
from imap_l3_processing.codice.l3.hi.models import PriorityEventL2, CodiceL2HiData, CodiceHiL2SectoredIntensitiesData, \
    CODICE_HI_NUM_L2_PRIORITIES
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.codice.l3.lo.constants import CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import NumpyArrayMatcher, get_test_instrument_team_data_path, get_test_data_path


class TestCodiceHiProcessor(unittest.TestCase):
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiL3aDirectEventsDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3a_direct_event")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    def test_process_l3a(self, mock_save_data, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')
        mock_processed_direct_events = Mock()
        mock_process_l3a.return_value = mock_processed_direct_events

        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_1')]

        processor = CodiceHiProcessor(input_collection, input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_direct_events)
        self.assertEqual([mock_save_data.return_value], product)
        self.assertEqual(['parent_file_1'], mock_processed_direct_events.parent_file_names)

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodicePitchAngleDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3b")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    def test_process_l3b_saves(self, mock_save_data, mock_process_l3b,
                               mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3b", start_date, end_date, 'v02')

        mock_processed_pitch_angles = Mock()
        mock_process_l3b.return_value = mock_processed_pitch_angles

        input_collection = MagicMock()
        input_collection.get_file_paths.return_value = [Path('path/to/parent_file_2')]

        processor = CodiceHiProcessor(input_collection, input_metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        mock_process_l3b.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_pitch_angles)
        self.assertEqual([mock_save_data.return_value], product)
        self.assertEqual(['parent_file_2'], mock_processed_pitch_angles.parent_file_names)

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02')

        processor = CodiceHiProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level for CoDICE: L2a", str(context.exception))

    def test_process_l3a_returns_data_product(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2025, 1, 1), datetime(2025, 1, 1)])
        epoch_delta_plus = np.full(epoch.shape, 1_000_000)

        num_epochs = len(epoch)
        event_buffer_size = 10

        l2_priority_events, (reshaped_l2_data_quality,
                             reshaped_l2_multi_flag,
                             _,
                             reshaped_l2_ssd_energy,
                             reshaped_l2_ssd_energy_plus,
                             reshaped_l2_ssd_energy_minus,
                             reshaped_l2_ssd_id,
                             reshaped_l2_spin_angle,
                             reshaped_l2_spin_number,
                             _,
                             reshaped_l2_type) = self._create_priority_events(num_epochs, event_buffer_size)

        expected_estimated_mass = np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size), np.nan)
        expected_energy_per_nuc = np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size), np.nan)

        reshaped_l2_num_events = np.empty((num_epochs, CODICE_HI_NUM_L2_PRIORITIES))
        reshaped_l2_time_of_flight = np.empty((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size))
        for priority_index, priority_event in enumerate(l2_priority_events):
            for epoch_i in range(num_epochs):
                number_of_events = rng.integers(0, 10)
                priority_event.number_of_events[epoch_i] = number_of_events

                time_of_flight_for_valid_events = rng.integers(1, 25, size=number_of_events)
                e_per_nuc_for_valid_events = time_of_flight_for_valid_events * 100
                ssd_energy_for_valid_events = priority_event.ssd_energy[epoch_i, :number_of_events]
                estimated_mass_for_valid_events = ssd_energy_for_valid_events / e_per_nuc_for_valid_events

                time_of_flight = np.full(event_buffer_size, 65535)
                time_of_flight[:number_of_events] = time_of_flight_for_valid_events

                priority_event.time_of_flight[epoch_i] = time_of_flight

                expected_energy_per_nuc[epoch_i, priority_index, :number_of_events] = e_per_nuc_for_valid_events
                expected_estimated_mass[epoch_i, priority_index, :number_of_events] = estimated_mass_for_valid_events

                reshaped_l2_num_events[epoch_i, priority_index] = number_of_events
                reshaped_l2_time_of_flight[epoch_i, priority_index, :] = time_of_flight

        l2_data = CodiceL2HiData(epoch, epoch_delta_plus, l2_priority_events)
        multiply_by_100_energy_per_nuc_lookup = TOFLookup(
            {i: EnergyPerNuc(i * 10, i * 100, i * 1000) for i in np.arange(1, 25)})
        dependencies = CodiceHiL3aDirectEventsDependencies(tof_lookup=multiply_by_100_energy_per_nuc_lookup,
                                                           codice_l2_hi_data=l2_data)

        processor = CodiceHiProcessor(Mock(), Mock())
        codice_direct_event_product = processor.process_l3a_direct_event(dependencies)

        np.testing.assert_array_equal(codice_direct_event_product.epoch, l2_data.epoch)
        np.testing.assert_array_equal(codice_direct_event_product.epoch_delta, l2_data.epoch_delta_plus)

        np.testing.assert_array_equal(codice_direct_event_product.data_quality, reshaped_l2_data_quality)
        np.testing.assert_array_equal(codice_direct_event_product.multi_flag, reshaped_l2_multi_flag)
        np.testing.assert_array_equal(codice_direct_event_product.num_events, reshaped_l2_num_events)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy, reshaped_l2_ssd_energy)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy_plus, reshaped_l2_ssd_energy_plus)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy_minus, reshaped_l2_ssd_energy_minus)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_id, reshaped_l2_ssd_id)
        np.testing.assert_array_equal(codice_direct_event_product.spin_angle,
                                      (reshaped_l2_spin_angle + CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM) % 360)
        np.testing.assert_array_equal(codice_direct_event_product.spin_number, reshaped_l2_spin_number)
        np.testing.assert_array_equal(codice_direct_event_product.tof, reshaped_l2_time_of_flight)
        np.testing.assert_array_equal(codice_direct_event_product.type, reshaped_l2_type)

        np.testing.assert_array_equal(codice_direct_event_product.estimated_mass, expected_estimated_mass)

        np.testing.assert_array_equal(expected_energy_per_nuc, codice_direct_event_product.energy_per_nuc)

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_unit_vector")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.get_sector_unit_vectors")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_pitch_angle")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.calculate_gyrophase")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.rebin_by_pitch_angle_and_gyrophase")
    def test_process_l3b_with_mocks(self, mock_rebin_by_pitch_angle_and_gyrophase, mock_calculate_gyrophases,
                                    mock_calculate_pitch_angles, mock_get_sector_unit_vectors,
                                    mock_calculate_unit_vector):
        rng = np.random.default_rng()
        epoch_1 = datetime(2025, 2, 5)
        epoch_2 = datetime(2025, 2, 6)
        epoch = np.array([epoch_1, epoch_2])
        epoch_delta = np.full(epoch.shape, timedelta(days=.5))

        rebinned_mag_data = rng.random((len(epoch), 3))
        mag_l1d_data = Mock()
        mag_l1d_data.rebin_to.return_value = rebinned_mag_data

        h_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17]) * 1.4), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        he3he4_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17]) * 1.4), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        cno_intensity = rng.random((len(epoch), len(np.array([1.11, 1.17, 1.25])), len(np.array([270, 15])), len(
            np.array([15, 45, 75, 105, 135, 165]))))
        fe_intensity = rng.random(
            (len(epoch), len(np.array([1.11, 1.17]) * 1.3), len(np.array([270, 15])),
             len(np.array([15, 45, 75, 105, 135, 165]))))

        codice_l2_data = CodiceHiL2SectoredIntensitiesData(
            epoch=epoch,
            epoch_delta_plus=epoch_delta,
            data_quality=sentinel.data_quality,
            elevation_angle=(np.array([270, 15])),
            spin_angles=(np.array([15, 45, 75, 105, 135, 165])),
            h_intensities=h_intensity,
            energy_h=(np.array([1.11, 1.17])),
            energy_h_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_h_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            cno_intensities=cno_intensity,
            energy_cno=(np.array([1.11, 1.17, 1.25])),
            energy_cno_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_cno_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            fe_intensities=fe_intensity,
            energy_fe=(np.array([1.11, 1.17]) * 1.3),
            energy_fe_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_fe_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            he3he4_intensities=he3he4_intensity,
            energy_he3he4=(np.array([1.11, 1.17]) * 1.4),
            energy_he3he4_plus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
            energy_he3he4_minus=(np.repeat(1.6, len(np.array([1.11, 1.17])))),
        )

        dependencies = CodicePitchAngleDependencies(mag_l1d_data=mag_l1d_data,
                                                    codice_sectored_intensities_data=codice_l2_data)

        expected_pitch_angles = np.linspace(15, 165, 6)
        expected_gyrophase = np.linspace(15, 345, 12)

        expected_pitch_angle_delta = np.repeat(15, 6)
        expected_gyrophase_delta = np.repeat(15, 12)

        sector_unit_vectors = rng.random((2, 6, 3))
        mag_unit_vectors = rng.random((2, 3))
        mock_calculate_unit_vector.side_effect = [mag_unit_vectors, sector_unit_vectors]

        mock_calculate_gyrophases.side_effect = [
            sentinel.epoch1_gyrophase,
            sentinel.epoch2_gyrophase,
        ]

        mock_calculate_pitch_angles.side_effect = [
            sentinel.epoch1_pitch_angle,
            sentinel.epoch2_pitch_angle,
        ]

        expected_h_intensity_binned_by_pa = rng.random((2, 2, 6))
        expected_he4_intensity_binned_by_pa = rng.random((2, 2, 6))
        expected_cno_intensity_binned_by_pa = rng.random((2, 3, 6))
        expected_fe_intensity_binned_by_pa = rng.random((2, 2, 6))

        expected_h_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))
        expected_he4_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))
        expected_cno_intensity_binned_by_pa_and_gyro = rng.random((2, 3, 6, 12))
        expected_fe_intensity_binned_by_pa_and_gyro = rng.random((2, 2, 6, 12))

        expected_parents = []

        mock_rebin_by_pitch_angle_and_gyrophase.side_effect = [
            (expected_h_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_h_intensity_binned_by_pa[0], 0, 0),
            (expected_he4_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_he4_intensity_binned_by_pa[0], 0, 0),
            (expected_cno_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_cno_intensity_binned_by_pa[0], 0, 0),
            (expected_fe_intensity_binned_by_pa_and_gyro[0], 0, 0, expected_fe_intensity_binned_by_pa[0], 0, 0),
            (expected_h_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_h_intensity_binned_by_pa[1], 0, 0),
            (expected_he4_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_he4_intensity_binned_by_pa[1], 0, 0),
            (expected_cno_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_cno_intensity_binned_by_pa[1], 0, 0),
            (expected_fe_intensity_binned_by_pa_and_gyro[1], 0, 0, expected_fe_intensity_binned_by_pa[1], 0, 0),
        ]

        codice_processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=sentinel.input_metadata)
        codice_hi_data_product = codice_processor.process_l3b(dependencies=dependencies)

        mock_get_sector_unit_vectors.assert_called_once_with(
            codice_l2_data.elevation_angle,
            NumpyArrayMatcher((codice_l2_data.spin_angles + CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM) % 360))
        mock_calculate_unit_vector.assert_has_calls(
            [call(NumpyArrayMatcher(rebinned_mag_data)), call(mock_get_sector_unit_vectors.return_value)])

        mock_calculate_pitch_angles.assert_has_calls([
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[0])),
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[1])),
        ])

        mock_calculate_gyrophases.assert_has_calls([
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[0])),
            call(NumpyArrayMatcher(-1 * sector_unit_vectors), NumpyArrayMatcher(mag_unit_vectors[1])),
        ])

        species_intensities = [h_intensity, he3he4_intensity, cno_intensity, fe_intensity]

        species_uncertainties = [
            np.zeros_like(h_intensity),
            np.zeros_like(he3he4_intensity),
            np.zeros_like(cno_intensity),
            np.zeros_like(fe_intensity),
        ]

        expected_rebin_calls = []
        for species_intensity, species_uncertainty in zip(species_intensities, species_uncertainties):
            expected_rebin_calls.append(
                call(intensity_data=NumpyArrayMatcher(species_intensity[0]),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty[0]),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty[0]),
                     pitch_angles=sentinel.epoch1_pitch_angle,
                     gyrophases=sentinel.epoch1_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     )

            )

        for species_intensity, species_uncertainty in zip(species_intensities, species_uncertainties):
            expected_rebin_calls.append(
                call(intensity_data=NumpyArrayMatcher(species_intensity[1]),
                     intensity_delta_plus=NumpyArrayMatcher(species_uncertainty[1]),
                     intensity_delta_minus=NumpyArrayMatcher(species_uncertainty[1]),
                     pitch_angles=sentinel.epoch2_pitch_angle,
                     gyrophases=sentinel.epoch2_gyrophase,
                     number_of_pitch_angle_bins=6, number_of_gyrophase_bins=12
                     )
            )

        mock_rebin_by_pitch_angle_and_gyrophase.assert_has_calls(expected_rebin_calls)

        self.assertEqual(sentinel.input_metadata, codice_hi_data_product.input_metadata)
        np.testing.assert_array_equal(epoch, codice_hi_data_product.epoch)
        np.testing.assert_array_equal(epoch_delta, codice_hi_data_product.epoch_delta)

        np.testing.assert_array_equal(codice_hi_data_product.energy_h, codice_l2_data.energy_h)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_plus, codice_l2_data.energy_h_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_h_minus, codice_l2_data.energy_h_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno, codice_l2_data.energy_cno)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_plus, codice_l2_data.energy_cno_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_cno_minus, codice_l2_data.energy_cno_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe, codice_l2_data.energy_fe)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_plus, codice_l2_data.energy_fe_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_fe_minus, codice_l2_data.energy_fe_minus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4, codice_l2_data.energy_he3he4)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_plus, codice_l2_data.energy_he3he4_plus)
        np.testing.assert_array_equal(codice_hi_data_product.energy_he3he4_minus, codice_l2_data.energy_he3he4_minus)

        np.testing.assert_array_almost_equal(expected_pitch_angles, codice_hi_data_product.pitch_angle)
        np.testing.assert_array_equal(expected_pitch_angle_delta, codice_hi_data_product.pitch_angle_delta)
        np.testing.assert_array_almost_equal(expected_gyrophase, codice_hi_data_product.gyrophase)
        np.testing.assert_array_equal(expected_gyrophase_delta, codice_hi_data_product.gyrophase_delta)

        np.testing.assert_allclose(codice_hi_data_product.h_intensity_by_pitch_angle, expected_h_intensity_binned_by_pa)

        np.testing.assert_array_equal(codice_hi_data_product.he3he4_intensity_by_pitch_angle,
                                      expected_he4_intensity_binned_by_pa)
        np.testing.assert_array_equal(codice_hi_data_product.cno_intensity_by_pitch_angle,
                                      expected_cno_intensity_binned_by_pa)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle,
                                      expected_fe_intensity_binned_by_pa)

        np.testing.assert_array_equal(codice_hi_data_product.h_intensity_by_pitch_angle_and_gyrophase,
                                      expected_h_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.he3he4_intensity_by_pitch_angle_and_gyrophase,
                                      expected_he4_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.cno_intensity_by_pitch_angle_and_gyrophase,
                                      expected_cno_intensity_binned_by_pa_and_gyro)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle_and_gyrophase,
                                      expected_fe_intensity_binned_by_pa_and_gyro)

        np.testing.assert_array_equal(codice_hi_data_product.parent_file_names, expected_parents)

    def test_integration_test(self):
        tof_lookup_path = get_test_instrument_team_data_path("codice/hi/imap_codice_tof-lookup_20241110_v002.csv")
        l2_direct_event_sci_path = get_test_data_path(
            "codice/imap_codice_l2_hi-direct-events_20241110_v002-all-fill.cdf")

        codice_hi_dependencies = CodiceHiL3aDirectEventsDependencies.from_file_paths(l2_direct_event_sci_path,
                                                                                     tof_lookup_path)

        input_metadata = InputMetadata(instrument='codice',
                                       data_level="l3a",
                                       start_date=Mock(spec=datetime),
                                       end_date=Mock(spec=datetime),
                                       version='v02',
                                       descriptor='hi-direct-events')
        processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=input_metadata)

        try:
            processor.process_l3a_direct_event(codice_hi_dependencies)
        except Exception as e:
            self.fail(e)

    def _create_priority_events(self, num_epochs, event_buffer_size):

        number_of_event_data_points = CODICE_HI_NUM_L2_PRIORITIES * num_epochs * event_buffer_size

        numbers = (np.arange(1, number_of_event_data_points + 1).reshape(CODICE_HI_NUM_L2_PRIORITIES, num_epochs,
                                                                         event_buffer_size))
        tof_all_events = (numbers % 24) + 1
        ssd_id_all_events = numbers * 1000

        (reshaped_l2_data_quality,
         reshaped_l2_number_of_events) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES), np.nan) for _ in range(2)]

        (reshaped_l2_multi_flag,
         reshaped_l2_ssd_energy,
         reshaped_l2_ssd_energy_plus,
         reshaped_l2_ssd_energy_minus,
         reshaped_l2_ssd_id,
         reshaped_l2_spin_angle,
         reshaped_l2_spin_number,
         reshaped_l2_time_of_flight,
         reshaped_l2_type) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size), np.nan) for _ in
                              range(9)]

        events = []
        for i in range(CODICE_HI_NUM_L2_PRIORITIES):
            data_quality = np.arange(num_epochs) * 1
            number_of_events = np.arange(num_epochs) + i

            ssd_energy = ssd_id_all_events[i]
            ssd_energy_plus = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i
            ssd_energy_minus = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i

            ssd_id = ssd_id_all_events[i]
            time_of_flight = tof_all_events[i]

            multi_flag = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i
            spin_angle = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            spin_number = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            type = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i

            events.append(PriorityEventL2(data_quality, multi_flag, number_of_events, ssd_energy, ssd_energy_plus,
                                          ssd_energy_minus, ssd_id,
                                          spin_angle, spin_number, time_of_flight, type))

            reshaped_l2_data_quality[:, i] = data_quality
            reshaped_l2_number_of_events[:, i] = number_of_events

            reshaped_l2_multi_flag[:, i, :] = multi_flag
            reshaped_l2_ssd_energy[:, i, :] = ssd_energy
            reshaped_l2_ssd_id[:, i, :] = ssd_id
            reshaped_l2_spin_angle[:, i, :] = spin_angle
            reshaped_l2_spin_number[:, i, :] = spin_number
            reshaped_l2_time_of_flight[:, i, :] = time_of_flight
            reshaped_l2_type[:, i, :] = type
            reshaped_l2_ssd_energy_plus[:, i, :] = ssd_energy_plus
            reshaped_l2_ssd_energy_minus[:, i, :] = ssd_energy_minus

        return events, (
            reshaped_l2_data_quality, reshaped_l2_multi_flag, reshaped_l2_number_of_events,
            reshaped_l2_ssd_energy, reshaped_l2_ssd_energy_plus, reshaped_l2_ssd_energy_minus, reshaped_l2_ssd_id,
            reshaped_l2_spin_angle, reshaped_l2_spin_number,
            reshaped_l2_time_of_flight, reshaped_l2_type)

    def _assert_estimated_mass(self, l2_priority_event, actual_calculated_mass, actual_energy_per_nuc,
                               expected_energy_per_nuc):
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc, actual_calculated_mass)
        np.testing.assert_array_equal(expected_energy_per_nuc, actual_energy_per_nuc)
