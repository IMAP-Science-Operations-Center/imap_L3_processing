import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call

from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.models import UpstreamDataDependency


class TestGlowsL3aDependencies(unittest.TestCase):

    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.CDF')
    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.download_dependency')
    def test_fetch_dependencies(self, mock_download_dependency, mock_cdf_constructor):
        with tempfile.TemporaryDirectory() as tmpdir:
            number_of_bins_path = Path(tmpdir, "number_of_bins.cdf")

            cases = [90, 200]
            for number_of_bins in cases:
                with self.subTest(number_of_bins=number_of_bins):
                    with open(number_of_bins_path, 'w') as f:
                        f.write(f"{number_of_bins}\n")

                    cdf_dependency = UpstreamDataDependency("glows", "l2a", datetime(2023, 1, 1), datetime(2023, 2, 1),
                                                            "1",
                                                            descriptor="histogram-01029")
                    dependency = UpstreamDataDependency("glows", "l2a", datetime(2023, 1, 1), datetime(2023, 2, 1), "1",
                                                        descriptor="not sci")
                    cdf_path_name = "some_cdf.cdf"

                    mock_download_dependency.side_effect = [
                        Path(cdf_path_name),
                        number_of_bins_path
                    ]

                    result = GlowsL3ADependencies.fetch_dependencies([dependency, cdf_dependency])

                    mock_cdf_constructor.assert_called_with(cdf_path_name)
                    self.assertIsInstance(result, GlowsL3ADependencies)
                    self.assertEqual(mock_cdf_constructor.return_value, result.data)
                    self.assertEqual(number_of_bins, result.number_of_bins)
                    expected_number_of_bins_dependency = UpstreamDataDependency("glows", "l2", None, None, "latest",
                                                                                "histogram-number-of-bins-text-not-cdf")
                    mock_download_dependency.assert_has_calls([
                        call(cdf_dependency),
                        call(expected_number_of_bins_dependency),
                    ])
