import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.lo.lo_processor import LoProcessor
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularIntensityDataProduct, \
    InputRectangularPointingSet
from imap_l3_processing.models import InputMetadata, Instrument


class TestLoProcessor(unittest.TestCase):

    @patch('imap_l3_processing.hi.hi_processor.MapProcessor.get_parent_file_names')
    @patch('imap_l3_processing.lo.lo_processor.LoL3SpectralFitDependencies.fetch_dependencies')
    @patch('imap_l3_processing.lo.lo_processor.fit_spectral_index_map')
    @patch('imap_l3_processing.lo.lo_processor.save_data')
    def test_process_spectral_index(self, mock_save_data,
                                    mock_fit_spectral_index_map, mock_fetch_dependencies,
                                    mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["some_input_file_name"]

        input_collection = Mock()
        lo_l3_spectral_fit_dependency = mock_fetch_dependencies.return_value
        lo_l3_spectral_fit_dependency.map_data.intensity_map_data.energy = np.array(
            [1, 10, 1000, 10000, 100000, 1000000, 10000000])
        mock_fit_spectral_index_map.return_value = Mock()

        metadata = InputMetadata(instrument="lo",
                                 data_level="l3",
                                 version="v000",
                                 start_date=datetime(2020, 1, 1, 1),
                                 end_date=datetime(2020, 1, 1, 1),
                                 descriptor="l090-spx-h-hf-sp-ram-hae-6deg-1yr")

        processor = LoProcessor(input_collection, input_metadata=metadata)
        product = processor.process()

        mock_fetch_dependencies.assert_called_with(input_collection)
        mock_fit_spectral_index_map.assert_called_once_with(lo_l3_spectral_fit_dependency.map_data.intensity_map_data)

        data_product = mock_save_data.call_args_list[0].args[0]

        self.assertIsInstance(data_product, RectangularSpectralIndexDataProduct)
        self.assertEqual(data_product.data.spectral_index_map_data,
                         mock_fit_spectral_index_map.return_value)
        self.assertEqual(data_product.data.coords, lo_l3_spectral_fit_dependency.map_data.coords)
        self.assertEqual(data_product.input_metadata, processor.input_metadata)
        self.assertEqual(data_product.parent_file_names, ["some_input_file_name"])
        self.assertEqual([mock_save_data.return_value], product)

    @patch('imap_l3_processing.lo.lo_processor.MapProcessor.get_parent_file_names')
    @patch("imap_l3_processing.lo.lo_processor.HiLoL3SurvivalDependencies.fetch_dependencies")
    @patch("imap_l3_processing.lo.lo_processor.process_survival_probabilities")
    @patch('imap_l3_processing.lo.lo_processor.save_data')
    def test_process_survival_probabilities(self, mock_save_data,
                                            mock_process_survival_prob,
                                            mock_fetch_survival_dependencies, mock_get_parent_file_names):
        mock_get_parent_file_names.return_value = ["somewhere"]
        cases = {
            "spacecraft": "sf",
            "heliospheric": "hf"
        }

        for case, reference_frame in cases.items():
            with self.subTest(case):
                input_metadata = InputMetadata(instrument="lo",
                                               data_level="l3",
                                               start_date=datetime.now(),
                                               end_date=datetime.now() + timedelta(days=1),
                                               version="",
                                               descriptor=f"l090-ena-h-{reference_frame}-sp-ram-hae-4deg-6mo",
                                               )

                dependencies = Mock(
                    l1c_data=[InputRectangularPointingSet(
                        epoch=datetime(2025, 1, 1),
                        epoch_delta=None,
                        repointing=1,
                        epoch_j2000=np.array([10]),
                        exposure_times=np.full((1, 7, 3600, 40), 2),
                        esa_energy_step=np.arange(7),
                        pointing_start_met=np.array([43200_000_000_000]),
                        pointing_end_met=np.array([43200_000_100_000]),
                    )],
                    dependency_file_paths=[Path("folder/map"), Path("folder/l1c")]
                )
                mock_fetch_survival_dependencies.return_value = dependencies

                mock_process_survival_prob.return_value = sentinel.survival_probabilities

                processor = LoProcessor(sentinel.input_dependencies, input_metadata)
                product = processor.process(spice_frame_name=SpiceFrame.IMAP_DPS)

                mock_fetch_survival_dependencies.assert_called_once_with(sentinel.input_dependencies,
                                                                         Instrument.IMAP_LO)

                mock_process_survival_prob.assert_called_once_with(dependencies, SpiceFrame.IMAP_DPS)

                np.testing.assert_array_equal(np.full((1, 7, 3600), 80), dependencies.l1c_data[0].exposure_times)

                mock_save_data.assert_called_once_with(RectangularIntensityDataProduct(
                    input_metadata=input_metadata,
                    parent_file_names=["l1c", "map", "somewhere"],
                    data=sentinel.survival_probabilities))
                self.assertEqual([mock_save_data.return_value], product)

                mock_fetch_survival_dependencies.reset_mock()
                mock_process_survival_prob.reset_mock()
                mock_save_data.reset_mock()

    def test_rejects_unimplemented_descriptors(self):
        input_collection = ProcessingInputCollection()

        cases = [
            ("not-a-valid-descriptor", ValueError, ("Could not parse descriptor not-a-valid-descriptor",)),
            ("l090-ena-h-hf-nsp-ram-hae-6deg-1yr", NotImplementedError, ("l090-ena-h-hf-nsp-ram-hae-6deg-1yr",)),
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

