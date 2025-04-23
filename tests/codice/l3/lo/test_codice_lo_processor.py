import unittest
from dataclasses import fields
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2bPriorityRates, \
    CodiceLoL2DirectEventData, CodiceLoL3aDirectEventDataProduct
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor
from tests.test_helpers import NumpyArrayMatcher


class TestCodiceLoProcessor(unittest.TestCase):
    def test_implements_processor(self):
        processor = CodiceLoProcessor(Mock(), Mock())
        self.assertIsInstance(processor, Processor)

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aDependencies.fetch_dependencies')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a')
    @patch(
        'imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor._process_l3a_direct_event_data_product')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    def test_process(self, mock_save_data, mock_process_direct_event, mock_process_l3a, mock_fetch_dependencies,
                     mock_upload):
        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')

        mock_save_data.side_effect = ["file1", "file2"]
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_process_direct_event.assert_called_once_with(mock_fetch_dependencies.return_value)

        mock_save_data.assert_has_calls(
            [call(mock_process_l3a.return_value), call(mock_process_direct_event.return_value)])

        mock_upload.assert_has_calls([call("file1"), call("file2")])

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_partial_densities')
    def test_process_l3a(self, mock_calculate_partial_densities):
        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)

        epochs = np.array([datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)])
        num_species = 13
        codice_lo_l2_data = Mock()
        codice_lo_l2_data.get_species_intensities = Mock()
        codice_lo_l2_data.get_species_intensities.return_value = {
            "H+": sentinel.h_intensities,
            "He++": sentinel.he_intensities,
            "C+4": sentinel.c4_intensities,
            "C+5": sentinel.c5_intensities,
            "C+6": sentinel.c6_intensities,
            "O+5": sentinel.o5_intensities,
            "O+6": sentinel.o6_intensities,
            "O+7": sentinel.o7_intensities,
            "O+8": sentinel.o8_intensities,
            "Mg": sentinel.mg_intensities,
            "Si": sentinel.si_intensities,
            "Fe (low Q)": sentinel.fe_low_intensities,
            "Fe (high Q)": sentinel.fe_high_intensities
        }
        codice_lo_l2_data.epoch = epochs

        mock_calculate_partial_densities.side_effect = [
            sentinel.h_partial_density,
            sentinel.he_partial_density,
            sentinel.c4_partial_density,
            sentinel.c5_partial_density,
            sentinel.c6_partial_density,
            sentinel.o5_partial_density,
            sentinel.o6_partial_density,
            sentinel.o7_partial_density,
            sentinel.o8_partial_density,
            sentinel.mg_partial_density,
            sentinel.si_partial_density,
            sentinel.fe_low_partial_density,
            sentinel.fe_high_partial_density,
        ]

        codice_lo_dependencies = CodiceLoL3aDependencies(codice_lo_l2_data, Mock(), Mock(), Mock(), Mock())
        result = processor.process_l3a(codice_lo_dependencies)

        self.assertEqual(num_species, mock_calculate_partial_densities.call_count)

        mock_calculate_partial_densities.assert_has_calls([call(sentinel.h_intensities),
                                                           call(sentinel.he_intensities),
                                                           call(sentinel.c4_intensities),
                                                           call(sentinel.c5_intensities),
                                                           call(sentinel.c6_intensities),
                                                           call(sentinel.o5_intensities),
                                                           call(sentinel.o6_intensities),
                                                           call(sentinel.o7_intensities),
                                                           call(sentinel.o8_intensities),
                                                           call(sentinel.mg_intensities),
                                                           call(sentinel.si_intensities),
                                                           call(sentinel.fe_low_intensities),
                                                           call(sentinel.fe_high_intensities)])
        self.assertIsInstance(result, CodiceLoL3aPartialDensityDataProduct)

        np.testing.assert_array_equal(result.epoch, np.full(1, np.nan))
        np.testing.assert_array_equal(result.epoch_delta, np.full(1, 4.8e+11))
        self.assertEqual(sentinel.h_partial_density, result.h_partial_density),
        self.assertEqual(sentinel.he_partial_density, result.he_partial_density),
        self.assertEqual(sentinel.c4_partial_density, result.c4_partial_density),
        self.assertEqual(sentinel.c5_partial_density, result.c5_partial_density),
        self.assertEqual(sentinel.c6_partial_density, result.c6_partial_density),
        self.assertEqual(sentinel.o5_partial_density, result.o5_partial_density),
        self.assertEqual(sentinel.o6_partial_density, result.o6_partial_density),
        self.assertEqual(sentinel.o7_partial_density, result.o7_partial_density),
        self.assertEqual(sentinel.o8_partial_density, result.o8_partial_density),
        self.assertEqual(sentinel.mg_partial_density, result.mg_partial_density),
        self.assertEqual(sentinel.si_partial_density, result.si_partial_density),
        self.assertEqual(sentinel.fe_low_partial_density, result.fe_low_partial_density),
        self.assertEqual(sentinel.fe_high_partial_density, result.fe_high_partial_density),

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_total_number_of_events')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_normalization_ratio')
    def test_process_l3a_direct_events(self, mock_calculate_normalization_ratio, mock_total_number_of_events):
        parameters = {f.name: None for f in fields(CodiceLoL2bPriorityRates)}

        priority_rates = CodiceLoL2bPriorityRates(
            **parameters,
        )
        epochs = np.array([datetime.now(), datetime.now() + timedelta(hours=1)])
        event_num = np.array([])

        rng = np.random.default_rng()
        priority_rates.epoch = epochs
        priority_rates.acquisition_times = rng.random((len(epochs), 2, 2))
        priority_rates.lo_sw_priority_p0_tcrs = rng.random((len(epochs), 2, 2))
        priority_rates.lo_sw_priority_p1_hplus = rng.random((len(epochs), 2, 2))
        priority_rates.lo_sw_priority_p2_heplusplus = rng.random((len(epochs), 2, 2))
        priority_rates.lo_sw_priority_p3_heavies = rng.random((len(epochs), 2, 2))
        priority_rates.lo_sw_priority_p4_dcrs = rng.random((len(epochs), 2, 2))
        priority_rates.lo_nsw_priority_p5_heavies = rng.random((len(epochs), 2, 2))
        priority_rates.lo_nsw_priority_p6_hplus_heplusplus = rng.random((len(epochs), 2, 2))
        priority_rates.lo_nsw_priority_p7_missing = rng.random((len(epochs), 2, 2))
        expected_total_events = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        expected_total_by_energy_and_spin_angle = [rng.random((128, 12)) for _ in range(16)]

        mock_total_number_of_events.side_effect = expected_total_events
        mock_calculate_normalization_ratio.side_effect = expected_total_by_energy_and_spin_angle

        priority_event_0 = Mock()
        priority_event_1 = Mock()
        priority_event_2 = Mock()
        priority_event_3 = Mock()
        priority_event_4 = Mock()
        priority_event_5 = Mock()
        priority_event_6 = Mock()
        priority_event_7 = Mock()

        priority_event_0.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_1.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_2.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_3.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_4.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_5.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_6.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]
        priority_event_7.total_events_binned_by_energy_step_and_spin_angle = [{} for _ in range(len(epochs))]

        direct_events = CodiceLoL2DirectEventData(epochs, event_num, priority_event_0, priority_event_1,
                                                  priority_event_2, priority_event_3, priority_event_4,
                                                  priority_event_5, priority_event_6, priority_event_7)

        dependencies = CodiceLoL3aDependencies(Mock(), priority_rates, direct_events, Mock(), Mock())

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        l3a_direct_event_data_product = processor._process_l3a_direct_event_data_product(dependencies)

        priority_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        self.assertIsInstance(l3a_direct_event_data_product, CodiceLoL3aDirectEventDataProduct)
        np.testing.assert_array_equal(epochs, l3a_direct_event_data_product.epoch)
        np.testing.assert_array_equal(priority_index, l3a_direct_event_data_product.priority)
        self.assertEqual(input_metadata, l3a_direct_event_data_product.input_metadata)

        self.assertEqual(8, mock_total_number_of_events.call_count)
        mock_total_number_of_events.assert_has_calls([
            call(NumpyArrayMatcher(priority_rates.lo_sw_priority_p0_tcrs),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_sw_priority_p1_hplus),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_sw_priority_p2_heplusplus),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_sw_priority_p3_heavies),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_sw_priority_p4_dcrs),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_nsw_priority_p5_heavies),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_nsw_priority_p6_hplus_heplusplus),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

            call(NumpyArrayMatcher(priority_rates.lo_nsw_priority_p7_missing),
                 NumpyArrayMatcher(priority_rates.acquisition_times)),

        ], any_order=False)

        expected_calls = []
        for index, priority_event in enumerate(direct_events.priority_events):
            epoch_call_1 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle[0],
                                expected_total_events[index][0])
            epoch_call_2 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle[0],
                                expected_total_events[index][1])
            expected_calls.extend([epoch_call_1, epoch_call_2])

            expected_epoch_1 = index * 2
            expected_epoch_2 = index * 2 + 1
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[0][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_1])
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[1][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_2])

        mock_calculate_normalization_ratio.assert_has_calls(expected_calls)


if __name__ == '__main__':
    unittest.main()
