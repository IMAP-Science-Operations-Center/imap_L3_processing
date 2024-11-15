import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.models import UpstreamDataDependency


class TestGlowsL3aDependencies(unittest.TestCase):

    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.CDF')
    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.download_dependency')
    def test_fetch_dependencies(self, mock_download_dependency, mock_cdf_constructor):
        cdf_dependency = UpstreamDataDependency("glows", "l2a", datetime(2023, 1, 1), datetime(2023, 2, 1), "1", descriptor="histogram-01029")
        dependency = UpstreamDataDependency("glows", "l2a", datetime(2023, 1, 1), datetime(2023, 2, 1), "1", descriptor="not sci")
        cdf_path_name = "some_cdf.cdf"
        number_of_bins = 90
        mock_download_dependency.return_value = Path(cdf_path_name)

        result = GlowsL3ADependencies.fetch_dependencies([dependency, cdf_dependency])

        mock_cdf_constructor.assert_called_with(cdf_path_name)
        self.assertIsInstance(result, GlowsL3ADependencies)
        self.assertEqual(mock_cdf_constructor.return_value, result.data)
        self.assertEqual(number_of_bins, result.number_of_bins)
        mock_download_dependency.assert_called_with(cdf_dependency)
