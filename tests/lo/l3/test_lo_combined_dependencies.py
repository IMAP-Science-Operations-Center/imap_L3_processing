import unittest
from unittest.mock import sentinel, call, patch

from imap_data_access import ScienceInput, ProcessingInputCollection

from imap_l3_processing.lo.l3.lo_combined_dependencies import LoCombinedDependencies
from imap_l3_processing.maps.map_models import RectangularIntensityMapData


class TestLoCombinedDependencies(unittest.TestCase):

    @patch("imap_l3_processing.lo.l3.lo_combined_dependencies.RectangularIntensityMapData.read_from_path")
    @patch("imap_l3_processing.lo.l3.lo_combined_dependencies.imap_data_access.download")
    def test_fetch_dependencies(self, mock_download, mock_read_from_path):
        lo75_input = 'imap_lo_l3_l075-ena-h-hf-sp-ram-hae-6deg-6mo_20250422_v001.cdf'
        lo90_input = 'imap_lo_l3_l090-ena-h-hf-sp-ram-hae-6deg-6mo_20250422_v001.cdf'
        lo105_input = 'imap_lo_l3_l105-ena-h-hf-sp-ram-hae-6deg-6mo_20250422_v001.cdf'

        mock_lo_map_data = [sentinel.lo75, sentinel.lo90, sentinel.lo105]
        mock_read_from_path.side_effect = mock_lo_map_data

        inputs = ProcessingInputCollection(ScienceInput(lo75_input), ScienceInput(lo90_input), ScienceInput(lo105_input))
        dependencies = LoCombinedDependencies.fetch_dependencies(inputs)

        mock_download.assert_has_calls(
            [
                call(lo75_input),
                call(lo90_input),
                call(lo105_input)
            ]
        )

        mock_read_from_path.assert_called_with(mock_download.return_value)
        self.assertEqual(mock_lo_map_data, dependencies.map_data)


