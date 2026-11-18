import logging
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ScienceFilePath,
    generate_imap_input,
)

import imap_l3_data_processor
from tests.integration.integration_test_helpers import mock_imap_data_access
from tests.test_helpers import get_run_local_data_path, get_test_data_path


class HitProcessorIntegration(unittest.TestCase):
    def _run_hit_l3_product(
        self,
        mock_parse_cli_arguments: Mock,
        input_files: list[Path],
        output_directory_name: str,
        descriptor: str,
        start_date: str,
    ) -> None:
        output_path = get_run_local_data_path(output_directory_name)

        with mock_imap_data_access(output_path, input_files):
            logging.basicConfig(
                force=True,
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

            processing_input_collection = ProcessingInputCollection(
                *(generate_imap_input(f.name) for f in input_files)
            )

            mock_arguments = Mock()
            mock_arguments.instrument = "hit"
            mock_arguments.data_level = "l3"
            mock_arguments.descriptor = descriptor
            mock_arguments.start_date = start_date
            mock_arguments.end_date = None
            mock_arguments.repointing = None
            mock_arguments.version = "v001"
            mock_arguments.dependency = processing_input_collection.serialize()
            mock_arguments.upload_to_sdc = False

            mock_parse_cli_arguments.return_value = mock_arguments

            imap_l3_data_processor.imap_l3_processor()

            expected_output_path = ScienceFilePath(
                f"imap_hit_l3_{descriptor}_{start_date}_v001.cdf"
            ).construct_path()
            self.assertTrue(
                expected_output_path.exists(),
                f"Expected file {expected_output_path.name} not found",
            )

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hit_macropixel_product(self, mock_parse_cli_arguments: Mock) -> None:
        input_files = [
            get_test_data_path("hit/imap_hit_l2_macropixel-intensity_20260228_v001.0002.cdf"),
            get_test_data_path("hit/imap_mag_l1d_norm-dsrf_20260228_v005.cdf"),
            get_test_data_path("hit/imap_mag_l2_norm-dsrf_20260228_v002.cdf"),
        ]
        self._run_hit_l3_product(
            mock_parse_cli_arguments,
            input_files,
            "hit_integration",
            "macropixel",
            "20260228",
        )

    @patch("imap_l3_data_processor._parse_cli_arguments")
    def test_hit_direct_events_product(self, mock_parse_cli_arguments: Mock) -> None:
        input_files = [
            get_test_data_path("hit/pha_events/imap_hit_l1a_direct-events_20100105_v009.cdf"),
            get_test_data_path("hit/pha_events/imap_hit_wincorr_20250901_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-2A-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-3A-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-4A-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-2B-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-3B-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-4B-cosine-lookup_20250203_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_lo-gain-lookup_20260514_v001.csv"),
            get_test_data_path("hit/pha_events/imap_hit_hi-gain-lookup_20250203_v000.csv"),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-2A-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-3A-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-4A-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-2B-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-3B-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path(
                "hit/pha_events/imap_hit_range-4B-charge-fit-lookup_20250319_v001.csv"
            ),
            get_test_data_path("hit/pha_events/imap_hit_hit-event-type-lookup_20250228_v003.csv"),
        ]
        self._run_hit_l3_product(
            mock_parse_cli_arguments,
            input_files,
            "hit_direct_events_integration",
            "direct-events",
            "20100105",
        )
