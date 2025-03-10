import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock, call

from imap_l3_processing.models import UpstreamDataDependency
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies, SWE_CONFIG_DESCRIPTOR


class TestSweL3Dependencies(unittest.TestCase):

    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.read_l1d_mag_data")
    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.read_l2_swe_data")
    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.read_l3a_swapi_proton_data")
    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.read_swe_config")
    def test_from_file_paths(self, mock_read_swe_config, mock_read_swapi_data, mock_read_swe_data, mock_read_mag_data):
        swe_path = Path("test_swe_cdf.cdf")
        mag_path = Path("test_mag_cdf.cdf")
        swapi_path = Path("test_swapi_cdf.cdf")
        config_path = Path("test_config.json")

        result = SweL3Dependencies.from_file_paths(swe_path, mag_path, swapi_path, config_path)

        mock_read_swe_data.assert_called_once_with(swe_path)
        mock_read_mag_data.assert_called_once_with(mag_path)
        mock_read_swapi_data.assert_called_once_with(swapi_path)
        mock_read_swe_config.assert_called_once_with(config_path)

        self.assertEqual(mock_read_swe_data.return_value, result.swe_l2_data)
        self.assertEqual(mock_read_mag_data.return_value, result.mag_l1d_data)
        self.assertEqual(mock_read_swapi_data.return_value, result.swapi_l3a_proton_data)
        self.assertEqual(mock_read_swe_config.return_value, result.configuration)

    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.download_dependency")
    @patch("imap_l3_processing.swe.l3.swe_l3_dependencies.SweL3Dependencies.from_file_paths")
    def test_fetch_dependencies(self, mock_from_file_paths, mock_download_dependency):
        swe_l2_dependency = UpstreamDataDependency("swe", "l2", datetime(2020, 1, 1), datetime(2020, 1, 1),
                                                   version="v0.00",
                                                   descriptor="sci")
        mag_l1d_dependency = UpstreamDataDependency("mag", "l1d", datetime(2020, 1, 1), datetime(2020, 1, 1),
                                                    version="v0.00",
                                                    descriptor="mago-normal")
        swapi_l3a_dependency = UpstreamDataDependency("swapi", "l3", datetime(2020, 1, 1), datetime(2020, 1, 1),
                                                      version="v0.00",
                                                      descriptor="proton-sw")

        expected_swe_path = Mock()
        expected_mag_path = Mock()
        expected_swapi_path = Mock()
        expected_config_path = Mock()
        mock_download_dependency.side_effect = [expected_swe_path, expected_mag_path, expected_swapi_path,
                                                expected_config_path]

        result = SweL3Dependencies.fetch_dependencies([swe_l2_dependency, mag_l1d_dependency, swapi_l3a_dependency])
        config_dependency = UpstreamDataDependency("swe", "l3", None, None, "latest",
                                                   SWE_CONFIG_DESCRIPTOR)

        mock_download_dependency.assert_has_calls([call(swe_l2_dependency),
                                                   call(mag_l1d_dependency),
                                                   call(swapi_l3a_dependency),
                                                   call(config_dependency)], any_order=False)

        mock_from_file_paths.assert_called_with(expected_swe_path, expected_mag_path, expected_swapi_path,
                                                expected_config_path)
        self.assertEqual(mock_from_file_paths.return_value, result)
