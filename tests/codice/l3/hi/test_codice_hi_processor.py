import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, sentinel, Mock

import numpy as np

from imap_l3_processing.codice.l3.hi.codice_hi_processor import CodiceHiProcessor
from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.direct_event.science.tof_lookup import TOFLookup, EnergyPerNuc
from imap_l3_processing.codice.l3.hi.models import PriorityEventL2, CodiceL2HiData, CodiceHiL2SectoredIntensitiesData, \
    CODICE_HI_NUM_L2_PRIORITIES
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.models import InputMetadata, MagL1dData


class TestCodiceHiProcessor(unittest.TestCase):
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiL3aDirectEventsDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3a_direct_event")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.upload")
    def test_process_l3a(self, mock_upload, mock_save_data, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')
        mock_processed_direct_events = Mock()
        mock_process_l3a.return_value = mock_processed_direct_events
        mock_expected_cdf = Mock()
        mock_save_data.return_value = mock_expected_cdf

        processor = CodiceHiProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(mock_processed_direct_events)
        mock_upload.assert_called_with(mock_expected_cdf)

    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodicePitchAngleDependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.CodiceHiProcessor.process_l3b")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.save_data")
    @patch("imap_l3_processing.codice.l3.hi.codice_hi_processor.upload")
    def test_process_l3b_saves_and_uploads(self, mock_upload, mock_save_data, mock_process_l3b,
                                           mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3b", start_date, end_date, 'v02')
        mock_process_l3b.return_value = sentinel.mock_processed_pitch_angles
        mock_expected_cdf = Mock()
        mock_save_data.return_value = mock_expected_cdf

        processor = CodiceHiProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3b.assert_called_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_with(sentinel.mock_processed_pitch_angles)
        mock_upload.assert_called_with(mock_expected_cdf)

    def test_raises_exception_on_non_l3_input_metadata(self):
        input_metadata = InputMetadata('codice', "L2a", Mock(), Mock(), 'v02')

        processor = CodiceHiProcessor(Mock(), input_metadata)
        with self.assertRaises(NotImplementedError) as context:
            processor.process()
        self.assertEqual("Unknown data level for CoDICE: L2a", str(context.exception))

    def test_process_l3a_returns_data_product(self):
        epochs = np.array([datetime(2025, 1, 1), datetime(2025, 1, 1)])

        l2_priority_events, (reshaped_l2_data_quality,
                             reshaped_l2_multi_flag,
                             reshaped_l2_number_of_events,
                             reshaped_l2_ssd_energy,
                             reshaped_l2_ssd_id,
                             reshaped_l2_spin_angle,
                             reshaped_l2_spin_number,
                             reshaped_l2_time_of_flight,
                             reshaped_l2_type) = self._create_priority_events()

        l2_data = CodiceL2HiData(epochs, l2_priority_events)
        multiply_by_100_energy_per_nuc_lookup = TOFLookup(
            {i: EnergyPerNuc(i * 10, i * 100, i * 1000) for i in np.arange(1, 25)})
        dependencies = CodiceHiL3aDirectEventsDependencies(tof_lookup=multiply_by_100_energy_per_nuc_lookup,
                                                           codice_l2_hi_data=l2_data)

        expected_energy_per_nuc = reshaped_l2_time_of_flight * 100

        processor = CodiceHiProcessor(Mock(), Mock())
        codice_direct_event_product = processor.process_l3a_direct_event(dependencies)

        np.testing.assert_array_equal(codice_direct_event_product.epoch, l2_data.epochs)
        np.testing.assert_array_equal(codice_direct_event_product.data_quality, reshaped_l2_data_quality)
        np.testing.assert_array_equal(codice_direct_event_product.multi_flag, reshaped_l2_multi_flag)
        np.testing.assert_array_equal(codice_direct_event_product.num_of_events, reshaped_l2_number_of_events)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy, reshaped_l2_ssd_energy)
        np.testing.assert_array_equal(codice_direct_event_product.ssd_id, reshaped_l2_ssd_id)
        np.testing.assert_array_equal(codice_direct_event_product.spin_angle, reshaped_l2_spin_angle)
        np.testing.assert_array_equal(codice_direct_event_product.spin_number, reshaped_l2_spin_number)
        np.testing.assert_array_equal(codice_direct_event_product.tof, reshaped_l2_time_of_flight)
        np.testing.assert_array_equal(codice_direct_event_product.type, reshaped_l2_type)

        np.testing.assert_array_equal(codice_direct_event_product.ssd_energy / expected_energy_per_nuc,
                                      codice_direct_event_product.estimated_mass)

        np.testing.assert_array_equal(expected_energy_per_nuc, codice_direct_event_product.energy_per_nuc)

    def test_process_l3b(self):
        rng = np.random.default_rng()
        epoch_1 = datetime(2025, 2, 5)
        epoch_2 = datetime(2025, 2, 6)
        mag_l1d_data = MagL1dData(
            epoch=np.array([epoch_1, epoch_1 + timedelta(days=0.49), epoch_2, epoch_2 + timedelta(days=0.49)]),
            mag_data=np.array(
                [
                    [.5, .25, .25], [.5, .25, .25],
                    [.5, .5, 0], [.5, .5, 0]
                ]),
        )
        spin_sector = np.array([60, 150])
        ssd_id = np.array([270, 15])

        epoch = np.array([epoch_1, epoch_2])
        epoch_delta = np.array([timedelta(days=.5), timedelta(days=.5)])
        energy = np.array([1.11, 1.17])
        energy_delta_minus = np.repeat(.4, len(energy))
        energy_delta_plus = np.repeat(1.6, len(energy))

        h_intensity = np.array([

            [np.arange(1, 5).reshape(2, 2),
             np.arange(1, 5).reshape(2, 2) * 2, ],
            [np.arange(6, 10).reshape(2, 2),
             np.arange(6, 10).reshape(2, 2) * 2, ]
        ])
        he4_intensity = rng.random((len(epoch), len(energy), len(ssd_id), len(spin_sector)))
        o_intensity = rng.random((len(epoch), len(energy), len(ssd_id), len(spin_sector)))
        fe_intensity = rng.random((len(epoch), len(energy), len(ssd_id), len(spin_sector)))

        expected_h_rebinned_pitch_angles = np.mean(h_intensity, axis=2)
        expected_he4_rebinned_pitch_angles = np.mean(he4_intensity, axis=2)
        expected_o_rebinned_pitch_angles = np.mean(o_intensity, axis=2)
        expected_fe_rebinned_pitch_angles = np.mean(fe_intensity, axis=2)

        codice_l2_data = CodiceHiL2SectoredIntensitiesData(
            epoch=epoch,
            epoch_delta=epoch_delta,
            energy=energy,
            ssd_id=ssd_id,
            spin_sector=spin_sector,
            h_intensities=h_intensity,
            he4_intensities=he4_intensity,
            o_intensities=o_intensity,
            fe_intensities=fe_intensity,
            energy_delta_minus=energy_delta_minus,
            energy_delta_plus=energy_delta_plus
        )

        dependencies = CodicePitchAngleDependencies(mag_l1d_data=mag_l1d_data,
                                                    codice_sectored_intensities_data=codice_l2_data)
        codice_processor = CodiceHiProcessor(dependencies=Mock(), input_metadata=sentinel.input_metadata)

        expected_pitch_angles = np.array([[81.406148, 138.590378], [56.104664, 115.658906]])
        expected_gyrophase = np.array([[26.890815, 49.106605], [291.063267, 286.102114]])

        expected_pitch_angle_delta = np.array([15, 15])
        expected_gyrophase_delta = np.array([30, 30])

        codice_hi_data_product = codice_processor.process_l3b(dependencies=dependencies)
        self.assertEqual(sentinel.input_metadata, codice_hi_data_product.input_metadata)
        np.testing.assert_array_equal(epoch, codice_hi_data_product.epoch)
        np.testing.assert_array_equal(epoch_delta, codice_hi_data_product.epoch_delta)
        np.testing.assert_array_equal(energy, codice_hi_data_product.energy)
        np.testing.assert_array_equal(energy_delta_plus, codice_hi_data_product.energy_delta_plus)
        np.testing.assert_array_equal(energy_delta_minus, codice_hi_data_product.energy_delta_minus)
        np.testing.assert_array_almost_equal(expected_pitch_angles, codice_hi_data_product.pitch_angle)
        np.testing.assert_array_equal(expected_pitch_angle_delta, codice_hi_data_product.pitch_angle_delta)
        np.testing.assert_array_almost_equal(expected_gyrophase, codice_hi_data_product.gyrophase)
        np.testing.assert_array_equal(expected_gyrophase_delta, codice_hi_data_product.gyrophase_delta)

        np.testing.assert_array_equal(expected_h_rebinned_pitch_angles,
                                      codice_hi_data_product.h_intensity_by_pitch_angle)
        np.testing.assert_array_equal(expected_he4_rebinned_pitch_angles,
                                      codice_hi_data_product.he4_intensity_by_pitch_angle)
        np.testing.assert_array_equal(expected_o_rebinned_pitch_angles,
                                      codice_hi_data_product.o_intensity_by_pitch_angle)
        np.testing.assert_array_equal(expected_fe_rebinned_pitch_angles,
                                      codice_hi_data_product.fe_intensity_by_pitch_angle)

        expected_h_rebinned_pitch_angles_and_gyrophase = np.transpose(h_intensity, (0, 1, 3, 2))
        expected_he4_rebinned_pitch_angles_and_gyrophase = np.transpose(he4_intensity, (0, 1, 3, 2))
        expected_o_rebinned_pitch_angles_and_gyrophase = np.transpose(o_intensity, (0, 1, 3, 2))
        expected_fe_rebinned_pitch_angles_and_gyrophase = np.transpose(fe_intensity, (0, 1, 3, 2))

        np.testing.assert_array_equal(codice_hi_data_product.h_intensity_by_pitch_angle_and_gyrophase,
                                      expected_h_rebinned_pitch_angles_and_gyrophase)

        np.testing.assert_array_equal(codice_hi_data_product.he4_intensity_by_pitch_angle_and_gyrophase,
                                      expected_he4_rebinned_pitch_angles_and_gyrophase)
        np.testing.assert_array_equal(codice_hi_data_product.o_intensity_by_pitch_angle_and_gyrophase,
                                      expected_o_rebinned_pitch_angles_and_gyrophase)
        np.testing.assert_array_equal(codice_hi_data_product.fe_intensity_by_pitch_angle_and_gyrophase,
                                      expected_fe_rebinned_pitch_angles_and_gyrophase)

    def _create_priority_events(self):
        num_epochs = 2
        event_buffer_size = 102

        number_of_event_data_points = CODICE_HI_NUM_L2_PRIORITIES * num_epochs * event_buffer_size

        numbers = (np.arange(1, number_of_event_data_points + 1).reshape(CODICE_HI_NUM_L2_PRIORITIES, num_epochs,
                                                                         event_buffer_size))
        tof_all_events = (numbers % 24) + 1
        ssd_id_all_events = numbers * 1000

        (reshaped_l2_data_quality,
         reshaped_l2_number_of_events) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES), np.nan) for _ in range(2)]

        (reshaped_l2_multi_flag,
         reshaped_l2_ssd_energy,
         reshaped_l2_ssd_id,
         reshaped_l2_spin_angle,
         reshaped_l2_spin_number,
         reshaped_l2_time_of_flight,
         reshaped_l2_type) = [np.full((num_epochs, CODICE_HI_NUM_L2_PRIORITIES, event_buffer_size), np.nan) for _ in
                              range(7)]

        events = []
        for i in range(CODICE_HI_NUM_L2_PRIORITIES):
            data_quality = np.arange(num_epochs) * 1
            number_of_events = np.arange(num_epochs) + i

            ssd_energy = ssd_id_all_events[i]
            ssd_id = ssd_id_all_events[i]
            time_of_flight = tof_all_events[i]

            multi_flag = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) + i
            spin_angle = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            spin_number = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i
            type = np.arange(num_epochs * event_buffer_size).reshape(num_epochs, event_buffer_size) * i

            events.append(PriorityEventL2(data_quality, multi_flag, number_of_events, ssd_energy, ssd_id,
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

        return events, (
            reshaped_l2_data_quality, reshaped_l2_multi_flag, reshaped_l2_number_of_events,
            reshaped_l2_ssd_energy, reshaped_l2_ssd_id, reshaped_l2_spin_angle, reshaped_l2_spin_number,
            reshaped_l2_time_of_flight, reshaped_l2_type)

    def _assert_estimated_mass(self, l2_priority_event, actual_calculated_mass, actual_energy_per_nuc,
                               expected_energy_per_nuc):
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc, actual_calculated_mass)
        np.testing.assert_array_equal(expected_energy_per_nuc, actual_energy_per_nuc)
