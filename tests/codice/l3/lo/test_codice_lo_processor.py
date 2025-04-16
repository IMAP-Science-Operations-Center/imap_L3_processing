import unittest
from datetime import datetime
from unittest.mock import Mock, patch, call, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.lo.codice_lo_l3a_dependencies import CodiceLoL3aDependencies
from imap_l3_processing.codice.l3.lo.codice_lo_processor import CodiceLoProcessor
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data, CodiceLoL3aDataProduct
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.esa_step_lookup import ESAStepLookup
from imap_l3_processing.codice.l3.lo.sectored_intensities.science.mass_per_charge_lookup import MassPerChargeLookup
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor


class TestCodiceLoProcessor(unittest.TestCase):
    def test_implements_processor(self):
        processor = CodiceLoProcessor(Mock(), Mock())
        self.assertIsInstance(processor, Processor)

    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.upload')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoL3aDependencies.fetch_dependencies')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.CodiceLoProcessor.process_l3a')
    @patch('imap_l3_processing.codice.l3.lo.codice_lo_processor.save_data')
    def test_process(self, mock_save_data, mock_process_l3a, mock_fetch_dependencies, mock_upload):
        input_collection = ProcessingInputCollection()
        input_metadata = InputMetadata('codice', "l3a", Mock(spec=datetime), Mock(spec=datetime), 'v02')

        processor = CodiceLoProcessor(dependencies=input_collection, input_metadata=input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_once_with(processor.dependencies)
        mock_process_l3a.assert_called_once_with(mock_fetch_dependencies.return_value)
        mock_save_data.assert_called_once_with(mock_process_l3a.return_value)
        mock_upload.assert_called_once_with(mock_save_data.return_value)

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

        codice_lo_dependencies = CodiceLoL3aDependencies(codice_lo_l2_data, Mock(), Mock())
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
        self.assertIsInstance(result, CodiceLoL3aDataProduct)

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


if __name__ == '__main__':
    unittest.main()
