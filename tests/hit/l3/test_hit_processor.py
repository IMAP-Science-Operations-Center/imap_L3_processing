import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import sentinel, patch, call, Mock

import numpy as np

from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.hit.l3.hit_processor import HITL3Processor, HIT_L2_DESCRIPTOR, MAG_L2_DESCRIPTOR
from imap_processing.hit.l3.models import HitL2Data
from imap_processing.models import UpstreamDataDependency, InputMetadata, MagL2Data
from tests.test_helpers import NumpyArrayMatcher


class TestHITL3Processor(TestCase):
    def setUp(self) -> None:
        self.temp_directory = f"{TEMP_CDF_FOLDER_PATH}"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

        self.mock_imap_patcher = patch('imap_processing.hit.l3.hit_processor.imap_data_access')
        self.mock_imap_api = self.mock_imap_patcher.start()
        self.mock_imap_api.query.side_effect = [
            [{'file_path': sentinel.data_file_path}],
            [{'file_path': sentinel.mag_file_path}]
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
        self.mock_imap_patcher.stop()

    @patch("imap_processing.hit.l3.hit_processor.CDF")
    @patch("imap_processing.hit.l3.hit_processor.read_l2_mag_data")
    @patch("imap_processing.hit.l3.hit_processor.read_l2_hit_data")
    @patch("imap_processing.hit.l3.hit_processor.calculate_unit_vector")
    @patch('imap_processing.hit.l3.hit_processor.utils')
    def test_processor(self, mock_utils, mock_calculate_unit_vector, mock_read_hit_data, mock_read_mag_data, mock_cdf_constructor):
        mock_cdf_constructor.side_effect = [sentinel.hit_cdf, sentinel.mag_cdf]

        hit_data = HitL2Data(
            epoch=np.array([datetime(2010, 1, 1, 0, 0, 46)]),
            epoch_delta=np.array([1800000000000]),
            flux=np.array([1, 2, 3, 4]),
            count_rates=np.array([5, 6, 7, 8]),
            uncertainty=np.array([2, 2, 2, 2, 2, 2, 2, 2]),
        )
        mag_data = MagL2Data(
            epoch=np.array([datetime(2010, 1, 1, 0, 0, 46), datetime(2010, 1, 1, 0, 1, 46), datetime(2010, 1, 1, 0, 2, 46)]),
            mag_data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        )
        mock_read_hit_data.return_value = hit_data
        mock_read_mag_data.return_value = mag_data

        upstream_l2_data_dependency = UpstreamDataDependency("hit", "l2",
                                                             datetime(2024, 9, 8),
                                                             datetime(2024, 9, 9),
                                                             "v001","let1-rates3600-fake-menlo")
        upstream_mag_data_dependency = UpstreamDataDependency("mag", "l2",
                                                              datetime(2024, 9, 6),
                                                              datetime(2024, 9, 7),
                                                              "v001", "fake-menlo-mag-SC-1min")

        data_processor_start = datetime(2024, 9, 10)
        data_processor_end = datetime(2024, 9, 11)

        input_metadata = InputMetadata("hit", "l3",
                                       data_processor_start,
                                       data_processor_end,
                                       "v001")
        hit_l3_data_processor = HITL3Processor([upstream_mag_data_dependency, upstream_l2_data_dependency],
                                               input_metadata)

        hit_data_path = Path()
        mag_data_path = Path()
        mock_utils.download_dependency.side_effect = [
            hit_data_path,
            mag_data_path
        ]

        hit_l3_data_processor.process()

        mock_utils.download_dependency.assert_has_calls([
            call(upstream_l2_data_dependency),
            call(upstream_mag_data_dependency)
        ])

        mock_cdf_constructor.assert_has_calls([
            call(str(hit_data_path)),
            call(str(mag_data_path))
        ])

        mock_read_hit_data.assert_called_with(sentinel.hit_cdf)
        mock_read_mag_data.assert_called_with(sentinel.mag_cdf)
        mock_calculate_unit_vector.assert_has_calls([
            call(NumpyArrayMatcher([0, 1, 2])),
            call(NumpyArrayMatcher([3, 4, 5])),
            call(NumpyArrayMatcher([6, 7, 8]))
        ])


    def test_throws_value_error_if_dependency_not_found(self):
        upstream_l2_data_dependency = UpstreamDataDependency("hit", "l2",
                                                             datetime(2024, 9, 8),
                                                             datetime(2024, 9, 9),
                                                             "v001","let1-rates3600-fake-menlo")
        upstream_mag_data_dependency = UpstreamDataDependency("mag", "l2-pre",
                                                              datetime(2024, 9, 6),
                                                              datetime(2024, 9, 7),
                                                              "v001", "fake-menlo-mag-SC-1min")

        input_metadata = InputMetadata("hit", "l3",
                                       datetime(2024, 9, 10),
                                       datetime(2024, 9, 11),
                                       "v001")

        cases = [(HIT_L2_DESCRIPTOR, [upstream_mag_data_dependency]), (MAG_L2_DESCRIPTOR, [upstream_l2_data_dependency])]
        for missing_descriptor, dependencies in cases:
            with self.subTest(missing_descriptor):
                hit_l3_data_processor = HITL3Processor(dependencies, input_metadata)

                with self.assertRaises(Exception) as cm:
                    hit_l3_data_processor.process()
                self.assertEqual(str(cm.exception), f"Missing {missing_descriptor} dependency.")