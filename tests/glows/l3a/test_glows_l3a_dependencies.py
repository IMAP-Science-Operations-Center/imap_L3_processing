import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, call

import numpy as np

from imap_processing.glows.descriptors import GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR, \
    GLOWS_NUMBER_OF_BINS_ANCILLARY_DESCRIPTOR, GLOWS_TIME_INDEPENDENT_BACKGROUND_DESCRIPTOR
from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.models import UpstreamDataDependency


class TestGlowsL3aDependencies(unittest.TestCase):

    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.TimeIndependentBackgroundLookupTable')
    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.CDF')
    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.download_dependency')
    @patch('imap_processing.glows.l3a.glows_l3a_dependencies.read_l2_glows_data')
    def test_fetch_dependencies(self, mock_read_l2_glows_data, mock_download_dependency, mock_cdf_constructor,
                                mock_time_independent_background_table_class):
        with tempfile.TemporaryDirectory() as tmpdir:
            number_of_bins_path = Path(tmpdir, "number_of_bins.cdf")
            background_path = Path(tmpdir, "background.cdf")
            with open(background_path, 'w') as f:
                f.write(f"4\n10\n6\n7\n")

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
                    time_independent_background_path = Path("/to/time/independent-background")

                    mock_download_dependency.side_effect = [
                        Path(cdf_path_name),
                        number_of_bins_path,
                        background_path,
                        time_independent_background_path,
                    ]

                    result = GlowsL3ADependencies.fetch_dependencies([dependency, cdf_dependency])

                    mock_cdf_constructor.assert_called_with(cdf_path_name)
                    mock_read_l2_glows_data.assert_called_with(mock_cdf_constructor.return_value)
                    self.assertIsInstance(result, GlowsL3ADependencies)
                    self.assertEqual(mock_read_l2_glows_data.return_value, result.data)
                    self.assertEqual(number_of_bins, result.number_of_bins)
                    self.assertEqual(mock_time_independent_background_table_class.from_file.return_value,
                                     result.time_independent_background_table)
                    np.testing.assert_equal([4, 10, 6, 7], result.time_dependent_background)
                    expected_number_of_bins_dependency = UpstreamDataDependency("glows", "l2", None, None, "latest",
                                                                                GLOWS_NUMBER_OF_BINS_ANCILLARY_DESCRIPTOR)
                    expected_time_dependent_background_dep = UpstreamDataDependency("glows", "l2", datetime(2023, 1, 1),
                                                                                    datetime(2023, 2, 1), "latest",
                                                                                    GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR)
                    expected_time_independent_background_dep = UpstreamDataDependency("glows", "l2",
                                                                                      None,
                                                                                      None, "latest",
                                                                                      GLOWS_TIME_INDEPENDENT_BACKGROUND_DESCRIPTOR)
                    mock_download_dependency.assert_has_calls([
                        call(cdf_dependency),
                        call(expected_number_of_bins_dependency),
                        call(expected_time_dependent_background_dep),
                        call(expected_time_independent_background_dep),
                    ])

                    mock_time_independent_background_table_class.from_file.assert_called_with(
                        time_independent_background_path)
