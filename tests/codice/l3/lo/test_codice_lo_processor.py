import unittest
from dataclasses import fields
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.models import CodiceLoL3aPartialDensityDataProduct, CodiceLoL2bPriorityRates, \
    CodiceLoL2DirectEventData, CodiceLoL3aDirectEventDataProduct, PriorityEvent
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

        codice_lo_dependencies = CodiceLoL3aDependencies(codice_lo_l2_data, Mock(), Mock(), Mock(), Mock(), Mock())
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
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_mass_per_charge')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.calculate_mass')
    def test_process_l3a_direct_events(self, mock_calculate_mass, mock_calculate_mass_per_charge,
                                       mock_calculate_normalization_ratio, mock_total_number_of_events):
        rng = np.random.default_rng()
        parameters = {f.name: None for f in fields(CodiceLoL2bPriorityRates)}

        priority_rates = CodiceLoL2bPriorityRates(
            **parameters,
        )

        epochs = np.array([datetime.now(), datetime.now() + timedelta(hours=1)])
        event_num = np.arange(10)

        priority_rates.epoch = epochs
        (priority_rates.acquisition_times,
         priority_rates.lo_sw_priority_p0_tcrs,
         priority_rates.lo_sw_priority_p1_hplus,
         priority_rates.lo_sw_priority_p2_heplusplus,
         priority_rates.lo_sw_priority_p3_heavies,
         priority_rates.lo_sw_priority_p4_dcrs,
         priority_rates.lo_nsw_priority_p5_heavies,
         priority_rates.lo_nsw_priority_p6_hplus_heplusplus,
         priority_rates.lo_nsw_priority_p7_missing) = [rng.random((len(epochs), 2, 2)) for _ in range(9)]

        expected_total_events = np.arange(1, 17).reshape(8, 2)
        expected_total_by_energy_and_spin_angle = [rng.random((128, 12)) for _ in range(16)]

        mass_per_charge_side_effect = [rng.random((len(epochs), len(event_num))) for _ in range(8)]
        mass_side_effect = [rng.random((len(epochs), len(event_num))) for _ in range(8)]

        (expected_mass,
         expected_mass_per_charge,
         expected_apd_energy,
         expected_apd_gain,
         expected_apd_id,
         expected_multi_flag,
         expected_pha_type,
         expected_tof) = [np.full((len(epochs), 8, len(event_num)), np.nan) for _ in range(8)]

        (expected_data_quality,
         expected_num_events) = [np.full((len(epochs), 8), np.nan) for _ in range(2)]

        mock_calculate_mass_per_charge.side_effect = mass_per_charge_side_effect
        mock_calculate_mass.side_effect = mass_side_effect
        mock_total_number_of_events.side_effect = expected_total_events
        mock_calculate_normalization_ratio.side_effect = expected_total_by_energy_and_spin_angle

        priority_events = []
        for i, (mass_per_charge, mass) in enumerate(
                zip(mass_per_charge_side_effect, mass_side_effect)):
            priority_event_kwargs = {f.name: rng.random((len(epochs), len(event_num))) for f in fields(PriorityEvent)}
            priority_event = PriorityEvent(**priority_event_kwargs)
            priority_event.data_quality = rng.random((len(epochs)))
            priority_event.num_events = rng.random((len(epochs)))
            priority_event.total_events_binned_by_energy_step_and_spin_angle = Mock()

            priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value = [{f"mock_p{i % 1}": i % 1}
                                                                                             for i in
                                                                                             range(len(epochs))]
            priority_events.append(priority_event)

            expected_apd_energy[:, i, :] = np.copy(priority_event.apd_energy)
            expected_apd_gain[:, i, :] = np.copy(priority_event.apd_gain)
            expected_apd_id[:, i, :] = np.copy(priority_event.apd_id)
            expected_multi_flag[:, i, :] = np.copy(priority_event.multi_flag)
            expected_pha_type[:, i, :] = np.copy(priority_event.pha_type)
            expected_tof[:, i, :] = np.copy(priority_event.tof)
            expected_mass[:, i, :] = np.copy(mass)
            expected_mass_per_charge[:, i, :] = np.copy(mass_per_charge)

            expected_data_quality[:, i] = np.copy(priority_event.data_quality)
            expected_num_events[:, i] = np.copy(priority_event.num_events)

        direct_events = CodiceLoL2DirectEventData(epochs, event_num, *priority_events)

        dependencies = CodiceLoL3aDependencies(Mock(), priority_rates, direct_events, Mock(), Mock(), Mock())

        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')
        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        l3a_direct_event_data_product = processor._process_l3a_direct_event_data_product(dependencies)

        priority_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        expected_calculate_normalization_calls = []
        expected_calculate_mass_calls = []
        expected_calculate_mass_per_charge_calls = []
        for index, priority_event in enumerate(direct_events.priority_events):
            epoch_call_1 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value[0],
                                expected_total_events[index][0])
            epoch_call_2 = call(priority_event.total_events_binned_by_energy_step_and_spin_angle.return_value[0],
                                expected_total_events[index][1])
            expected_calculate_normalization_calls.extend([epoch_call_1, epoch_call_2])
            expected_calculate_mass_calls.append(call(priority_event, dependencies.mass_coefficient_lookup))
            expected_calculate_mass_per_charge_calls.append(call(priority_event))

            expected_epoch_1 = index * 2
            expected_epoch_2 = index * 2 + 1
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[0][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_1])
            np.testing.assert_array_equal(l3a_direct_event_data_product.normalization[1][index],
                                          expected_total_by_energy_and_spin_angle[expected_epoch_2])

        self.assertIsInstance(l3a_direct_event_data_product, CodiceLoL3aDirectEventDataProduct)
        self.assertEqual(input_metadata, l3a_direct_event_data_product.input_metadata)

        np.testing.assert_array_equal(epochs, l3a_direct_event_data_product.epoch)
        np.testing.assert_array_equal(priority_index, l3a_direct_event_data_product.priority)
        np.testing.assert_array_equal(expected_mass_per_charge, l3a_direct_event_data_product.mass_per_charge)
        np.testing.assert_array_equal(expected_mass, l3a_direct_event_data_product.mass)

        mock_calculate_normalization_ratio.assert_has_calls(expected_calculate_normalization_calls)
        mock_calculate_mass.assert_has_calls(expected_calculate_mass_calls)

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

        np.testing.assert_array_equal(expected_apd_energy, l3a_direct_event_data_product.energy)
        np.testing.assert_array_equal(expected_apd_gain, l3a_direct_event_data_product.gain)
        np.testing.assert_array_equal(expected_apd_id, l3a_direct_event_data_product.apd_id)
        np.testing.assert_array_equal(expected_multi_flag, l3a_direct_event_data_product.multi_flag)
        np.testing.assert_array_equal(expected_num_events, l3a_direct_event_data_product.num_events)
        np.testing.assert_array_equal(expected_data_quality, l3a_direct_event_data_product.data_quality)
        np.testing.assert_array_equal(event_num, l3a_direct_event_data_product.event_num)
        np.testing.assert_array_equal(expected_pha_type, l3a_direct_event_data_product.pha_type)
        np.testing.assert_array_equal(expected_tof, l3a_direct_event_data_product.tof)


if __name__ == '__main__':
    unittest.main()
