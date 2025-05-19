import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularIntensityDataProduct
from imap_l3_processing.models import InputMetadata, Instrument


class TestLoProcessor(unittest.TestCase):

    @patch('imap_l3_processing.hi.hi_processor.Processor.get_parent_file_names')
    @patch('imap_l3_processing.lo.lo_processor.LoL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.lo.lo_processor.calculate_spectral_index_for_multiple_ranges')
    @patch('imap_l3_processing.lo.lo_processor.save_data')
    @patch('imap_l3_processing.lo.lo_processor.upload')
    def test_process_spectral_index(self, mock_upload, mock_save_data,
                                    mock_calculate_spectral_index_for_multiple_ranges, mock_fetch_dependencies,
                                    mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["some_input_file_name"]

        input_collection = Mock()
        lo_l3_spectral_fit_dependency = mock_fetch_dependencies.return_value
        lo_l3_spectral_fit_dependency.map_data.intensity_map_data.energy = np.array(
            [1, 10, 1000, 10000, 100000, 1000000, 10000000])
        mock_calculate_spectral_index_for_multiple_ranges.return_value = Mock()

        metadata = InputMetadata(instrument="lo",
                                 data_level="l3",
                                 version="v000",
                                 start_date=datetime(2020, 1, 1, 1),
                                 end_date=datetime(2020, 1, 1, 1),
                                 descriptor="l090-spx-h-hf-sp-ram-hae-6deg-1yr")

        processor = LoProcessor(input_collection, input_metadata=metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        energy_range = (10000, np.inf)
        mock_calculate_spectral_index_for_multiple_ranges.assert_called_once_with(
            lo_l3_spectral_fit_dependency.map_data.intensity_map_data, [
                energy_range])

        data_product = mock_save_data.call_args_list[0].args[0]

        self.assertIsInstance(data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(data_product.data.spectral_index_map_data,
                         mock_calculate_spectral_index_for_multiple_ranges.return_value)
        self.assertEqual(data_product.data.coords, lo_l3_spectral_fit_dependency.map_data.coords)
        self.assertEqual(data_product.input_metadata, processor.input_metadata)
        self.assertEqual(data_product.parent_file_names, ["some_input_file_name"])
        mock_upload.assert_called_with(mock_save_data.return_value)

    @patch('imap_l3_processing.lo.lo_processor.Processor.get_parent_file_names')
    @patch("imap_l3_processing.lo.lo_processor.HiLoL3SurvivalDependencies.fetch_dependencies")
    @patch("imap_l3_processing.lo.lo_processor.process_survival_probabilities")
    @patch('imap_l3_processing.lo.lo_processor.save_data')
    @patch('imap_l3_processing.lo.lo_processor.upload')
    def test_process_survival_probabilities(self, mock_upload, mock_save_data,
                                            mock_process_survival_prob,
                                            mock_fetch_survival_dependencies, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["somewhere"]

        mock_upload.reset_mock()
        mock_save_data.reset_mock()
        mock_fetch_survival_dependencies.reset_mock()
        mock_process_survival_prob.reset_mock()

        input_metadata = InputMetadata(instrument="lo",
                                       data_level="l3",
                                       start_date=datetime.now(),
                                       end_date=datetime.now() + timedelta(days=1),
                                       version="",
                                       descriptor="l090-ena-h-sf-sp-ram-hae-4deg-6mo",
                                       )

        dependencies: HiLoL3SurvivalDependencies = mock_fetch_survival_dependencies.return_value
        dependencies.dependency_file_paths = [Path("folder/map"), Path("folder/l1c")]

        mock_process_survival_prob.return_value = sentinel.survival_probabilities

        processor = LoProcessor(sentinel.input_dependencies, input_metadata)
        processor.process()

        mock_fetch_survival_dependencies.assert_called_once_with(sentinel.input_dependencies,
                                                                 Instrument.IMAP_LO)

        mock_process_survival_prob.assert_called_once_with(dependencies)

        mock_save_data.assert_called_once_with(RectangularIntensityDataProduct(
            input_metadata=input_metadata,
            parent_file_names=["l1c", "map", "somewhere"],
            data=sentinel.survival_probabilities))
        mock_upload.assert_called_once_with(mock_save_data.return_value)

    def test_rejects_unimplemented_descriptors(self):
        input_collection = ProcessingInputCollection()

        cases = [
            ("not-a-valid-descriptor", ValueError, ("Could not parse descriptor not-a-valid-descriptor",)),
            ("l090-ena-h-hf-sp-ram-hae-6deg-1yr", NotImplementedError, ("l090-ena-h-hf-sp-ram-hae-6deg-1yr",)),
        ]
        for descriptor, exception_class, exception_args in cases:
            with self.subTest(descriptor):
                metadata = InputMetadata(instrument="lo",
                                         data_level="l3",
                                         version="v000",
                                         start_date=datetime(2020, 1, 1, 1),
                                         end_date=datetime(2020, 1, 1, 1),
                                         descriptor=descriptor)

                processor = LoProcessor(input_collection, input_metadata=metadata)
                with self.assertRaises(exception_class) as cm:
                    processor.process()
                self.assertEqual(exception_args, cm.exception.args)
