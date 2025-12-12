import unittest
from datetime import datetime
from unittest.mock import patch, call, Mock

from imap_l3_processing.hi.hi_combined_initializer import HiCombinedL3Initializer
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import create_mock_query_results


class TestHiCombinedL3Initializer(unittest.TestCase):
    def setUp(self):
        self.map_initializer_query_patcher = patch('imap_l3_processing.maps.map_initializer.imap_data_access.query')
        self.map_initializer_mock_query = self.map_initializer_query_patcher.start()

    def teardown(self):
        self.map_initializer_mock_query.stop()

    def test_get_maps_that_should_be_produced_no_existing_combined(self):
        self.map_initializer_mock_query.side_effect = [
            create_mock_query_results([
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',

                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',

                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',

                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',
            ]),
            create_mock_query_results([])
        ]

        expected_maps_to_produce = [
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                    'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',

                    'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                    'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument='hi',
                    data_level='l3',
                    start_date=datetime(2025, 1, 1),
                    end_date=datetime(2026, 1, 1),
                    version='v001',
                    descriptor='hic-ena-h-hf-sp-full-hae-6deg-1yr'
                )
            )
        ]

        initializer = HiCombinedL3Initializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced('hic-ena-h-hf-sp-full-hae-6deg-1yr')

        self.map_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            # call(instrument='hi', data_level='l2'),
        ])

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_existing_combined_and_6month_but_no_new_maps(self,
                                                                                           mock_read_cdf_parents: Mock):
        self.map_initializer_mock_query.side_effect = [
            create_mock_query_results([
                'imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',

                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20260101_v001.cdf',

                'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',

                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20260101_v001.cdf',

                'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',
            ]),
            create_mock_query_results([]),
        ]

        mock_read_cdf_parents.return_value = {
            'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
            'imap_hi_l3_h45-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',
            'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250101_v001.cdf',
            'imap_hi_l3_h90-ena-h-hf-sp-full-hae-6deg-6mo_20250701_v001.cdf',
        }

        expected_maps_to_produce = []

        initializer = HiCombinedL3Initializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced('hic-ena-h-hf-sp-full-hae-6deg-1yr')

        self.map_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            # call(instrument='hi', data_level='l2'),
        ])

        mock_read_cdf_parents.assert_called_once_with("imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf")

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)
