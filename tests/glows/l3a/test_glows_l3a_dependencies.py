import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call

from imap_l3_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestGlowsL3aDependencies(unittest.TestCase):

    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.download_dependency_with_repointing')
    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.CDF')
    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.download_dependency')
    @patch('imap_l3_processing.glows.l3a.glows_l3a_dependencies.read_l2_glows_data')
    def test_fetch_dependencies(self, mock_read_l2_glows_data, mock_download_dependency, mock_cdf_constructor,
                                mock_download_dependency_with_repointing):
        ignored_dependency = UpstreamDataDependency("glows", "l2b", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                    "1",
                                                    descriptor="ignored-data")

        cdf_dependency = UpstreamDataDependency("glows", "l2a", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                "1",
                                                descriptor="histogram-01029")
        calibration_data_dependency = UpstreamDataDependency("glows", "l3a", None, None, "latest",
                                                             descriptor="calibration-data-text-not-cdf")
        settings_dependency = UpstreamDataDependency("glows", "l3a", None, None, "latest",
                                                     descriptor="pipeline-settings-json-not-cdf")
        extra_heliospheric_background_dependency = UpstreamDataDependency("glows", "l3a",
                                                                          None, None, "latest",
                                                                          descriptor="map-of-extra-helio-bckgrd-text-not-cdf")
        time_dependent_background_dependency = UpstreamDataDependency("glows", "l3a",
                                                                      None, None, "latest",
                                                                      descriptor="time-dep-bckgrd-text-not-cdf")

        cdf_path_str = "some_cdf.cdf"
        cdf_path_name = Path(cdf_path_str)
        calibration_data = Path("calibration.dat")
        settings = Path("settings.dat")
        time_dependent_background_path = Path("time_dependent_background.dat")
        extra_heliospheric_background = Path("extra_heliospheric_background.dat")

        mock_download_dependency.side_effect = [
            calibration_data,
            extra_heliospheric_background,
            time_dependent_background_path,
            settings,
        ]

        mock_download_dependency_with_repointing.return_value = (cdf_path_name, 2)

        result = GlowsL3ADependencies.fetch_dependencies([ignored_dependency, cdf_dependency])

        mock_cdf_constructor.assert_called_with(cdf_path_str)
        mock_read_l2_glows_data.assert_called_with(mock_cdf_constructor.return_value)
        self.assertIsInstance(result, GlowsL3ADependencies)
        self.assertEqual(mock_read_l2_glows_data.return_value, result.data)

        self.assertEqual(calibration_data, result.ancillary_files["calibration_data"])
        self.assertEqual(settings, result.ancillary_files["settings"])
        self.assertEqual(time_dependent_background_path,
                         result.ancillary_files["time_dependent_bckgrd"])
        self.assertEqual(extra_heliospheric_background,
                         result.ancillary_files["extra_heliospheric_bckgrd"])
        self.assertEqual(2, result.repointing)

        mock_download_dependency_with_repointing.assert_called_once_with(cdf_dependency)

        self.assertEqual([
            call(calibration_data_dependency),
            call(extra_heliospheric_background_dependency),
            call(time_dependent_background_dependency),
            call(settings_dependency),
        ], mock_download_dependency.call_args_list)
