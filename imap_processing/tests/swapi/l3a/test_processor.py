from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, sentinel

import imap_processing
from imap_processing.models import UpstreamDataDependency
from imap_processing.swapi.l3a.processor import SwapiL3AProcessor


class TestProcessor(TestCase):
    @patch('imap_processing.swapi.l3a.processor.imap_data_access')
    def test_processor(self, mock_imap_api):
        file_path = Path(
            imap_processing.__file__).parent.parent / 'swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v002.cdf'

        mock_imap_api.query.return_value = [{'file_path':sentinel.file_path}]
        mock_imap_api.download.return_value = file_path
        swapi_processor = SwapiL3AProcessor([UpstreamDataDependency('swapi', 'l2', 'c', datetime.now(), datetime.now(), 'f')], "swapi", "l3a", datetime.now()-timedelta(days=1), datetime.now(), "12345")

        swapi_processor.process()