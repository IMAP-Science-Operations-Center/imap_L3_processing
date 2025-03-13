import unittest
from pathlib import Path
from unittest.mock import patch

from imap_l3_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies
from imap_l3_processing.models import UpstreamDataDependency


class TestHiL3Dependencies(unittest.TestCase):

    @patch("imap_l3_processing.hi.l3.hi_l3_dependencies.read_hi_l3_data")
    def test_from_file_paths(self, read_hi_l3_data):
        hi_l3 = Path("test_hi_l3_cdf.cdf")

        result = HiL3Dependencies.from_file_paths(hi_l3)

        read_hi_l3_data.assert_called_with(hi_l3)

        self.assertEqual(read_hi_l3_data.call_count, 1)
        self.assertEqual(read_hi_l3_data.return_value, result.hi_l3_data)

    @patch("imap_l3_processing.hi.l3.hi_l3_dependencies.download_dependency")
    @patch("imap_l3_processing.hi.l3.hi_l3_dependencies.read_hi_l3_data")
    def test_fetch_dependencies(self, mock_read_hi_l3_data, mock_download_dependency):
        hi_l3_dependency = UpstreamDataDependency("hi", "l3", None, None, "latest", "spectral-fit-index")
        dependencies = [hi_l3_dependency]

        dependency = HiL3Dependencies.fetch_dependencies(dependencies)

        mock_download_dependency.assert_called_with(hi_l3_dependency)
        mock_read_hi_l3_data.assert_called_with(mock_download_dependency.return_value)
        self.assertEqual(mock_read_hi_l3_data.return_value, dependency.hi_l3_data)

    @patch("imap_l3_processing.hi.l3.hi_l3_dependencies.download_dependency")
    def test_raises_value_error_if_instrument_doesnt_match(self, _):
        hi_l3_dependency = UpstreamDataDependency("bad", "l3", None, None, "latest", "hi_l3_data")
        dependencies = [hi_l3_dependency]

        with self.assertRaises(ValueError) as error:
            _ = HiL3Dependencies.fetch_dependencies(dependencies)

        self.assertEqual(str(error.exception), "Missing Hi dependency.")
