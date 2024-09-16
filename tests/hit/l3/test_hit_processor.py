import os
import shutil
from datetime import datetime
from unittest import TestCase
from unittest.mock import sentinel, patch, call

from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.hit.l3.hit_processor import HITL3Processor
from imap_processing.models import UpstreamDataDependency, InputMetadata


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

    def test_processor(self):
        upstream_l2_data_dependency = UpstreamDataDependency("hit", "l2",
                                                             datetime(2024, 9, 8),
                                                             datetime(2024, 9, 9),
                                                             "v001","HIT_Flux_and_Count_Rate")
        upstream_mag_data_dependency = UpstreamDataDependency("mag", "l2-pre",
                                                              datetime(2024, 9, 6),
                                                              datetime(2024, 9, 7),
                                                              "v001", "mag_field")

        data_processor_start = datetime(2024, 9, 10)
        data_processor_end = datetime(2024, 9, 11)

        input_metadata = InputMetadata("hit", "l3",
                                       data_processor_start,
                                       data_processor_end,
                                       "v001")
        hit_l3_data_processor = HITL3Processor([upstream_l2_data_dependency, upstream_mag_data_dependency],
                                               input_metadata)

        hit_l3_data_processor.process()

        self.mock_imap_api.query.assert_has_calls([
            call(
                instrument=upstream_l2_data_dependency.instrument,
                data_level=upstream_l2_data_dependency.data_level,
                descriptor=upstream_l2_data_dependency.descriptor,
                start_date="20240910",
                end_date="20240911",
                version="latest"
            ),
            call(
                instrument=upstream_mag_data_dependency.instrument,
                data_level=upstream_mag_data_dependency.data_level,
                descriptor=upstream_mag_data_dependency.descriptor,
                start_date="20240910",
                end_date="20240911",
                version="latest"
            )
        ])
