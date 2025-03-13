from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call
from imap_l3_processing.models import UpstreamDataDependency, InputMetadata

from imap_l3_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies, MAG_L1D_DESCRIPTOR, \
    HIT_L2_DESCRIPTOR


class TestHITL3SectoredDependencies(TestCase):
    @patch("imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.read_l1d_mag_data")
    @patch("imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.read_l2_hit_data")
    @patch('imap_l3_processing.hit.l3.hit_l3_sectored_dependencies.download_dependency')
    def test_fetch_dependencies(self, mock_download_dependency, mock_read_hit_data, mock_read_mag_data):
        upstream_l2_data_dependency = UpstreamDataDependency("hit", "l2",
                                                             datetime(2024, 9, 8),
                                                             datetime(2024, 9, 9),
                                                             "v001", HIT_L2_DESCRIPTOR)
        upstream_mag_data_dependency = UpstreamDataDependency("mag", "l2",
                                                              datetime(2024, 9, 6),
                                                              datetime(2024, 9, 7),
                                                              "v001", MAG_L1D_DESCRIPTOR)

        hit_data_path = Path("hit")
        mag_data_path = Path("mag")
        mock_download_dependency.side_effect = [
            hit_data_path,
            mag_data_path
        ]

        deps = HITL3SectoredDependencies.fetch_dependencies([upstream_mag_data_dependency, upstream_l2_data_dependency])

        mock_download_dependency.assert_has_calls([
            call(upstream_l2_data_dependency),
            call(upstream_mag_data_dependency)
        ])

        mock_read_hit_data.assert_called_with(hit_data_path)
        mock_read_mag_data.assert_called_with(mag_data_path)

        self.assertEqual(mock_read_hit_data.return_value, deps.data)
        self.assertEqual(mock_read_mag_data.return_value, deps.mag_l1d_data)

    def test_throws_value_error_if_dependency_not_found(self):
        upstream_l2_data_dependency = UpstreamDataDependency("hit", "l2",
                                                             datetime(2024, 9, 8),
                                                             datetime(2024, 9, 9),
                                                             "v001", HIT_L2_DESCRIPTOR)
        upstream_mag_data_dependency = UpstreamDataDependency("mag", "l2-pre",
                                                              datetime(2024, 9, 6),
                                                              datetime(2024, 9, 7),
                                                              "v001", MAG_L1D_DESCRIPTOR)

        cases = [(HIT_L2_DESCRIPTOR, [upstream_mag_data_dependency]),
                 (MAG_L1D_DESCRIPTOR, [upstream_l2_data_dependency])]
        for missing_descriptor, dependencies in cases:
            with self.subTest(missing_descriptor):
                with self.assertRaises(Exception) as cm:
                    fetched = HITL3SectoredDependencies.fetch_dependencies(dependencies)
                self.assertEqual(str(cm.exception), f"Missing {missing_descriptor} dependency.")
