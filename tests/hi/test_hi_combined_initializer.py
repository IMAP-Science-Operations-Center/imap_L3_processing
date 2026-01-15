import unittest
from datetime import datetime
from unittest.mock import patch, Mock, call

from imap_l3_processing.hi.hi_combined_initializer import HiCombinedInitializer
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import create_mock_query_results

l3_sp_input_maps = [
    'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',
]

l2_nsp_input_maps = [
    'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-6deg-1yr_20250101_v001.cdf',
    'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-6deg-1yr_20250101_v001.cdf',
]


class TestHiCombinedInitializer(unittest.TestCase):
    def setUp(self):
        self.hi_combined_initializer_query_patcher = patch(
            'imap_l3_processing.hi.hi_combined_initializer.imap_data_access.query')
        self.hi_combined_initializer_mock_query = self.hi_combined_initializer_query_patcher.start()

    def teardown(self):
        self.hi_combined_initializer_mock_query.stop()

    def test_get_maps_that_should_be_produced_no_existing_combined_sp(self):
        for deg in ["6deg", "4deg"]:
            with self.subTest(msg=f"'hic-ena-h-hf-sp-full-hae-{deg}-1yr'"):
                self.hi_combined_initializer_mock_query.side_effect = [
                    create_mock_query_results([
                        f'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                        f'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-{deg}-1yr_20250101_v001.cdf',

                        f'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                        f'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-{deg}-1yr_20250101_v001.cdf',
                    ]),
                    create_mock_query_results([])
                ]

                expected_maps_to_produce = [
                    PossibleMapToProduce(
                        input_files={
                            f'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                            f'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-{deg}-1yr_20250101_v001.cdf',

                            f'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                            f'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-{deg}-1yr_20250101_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument='hi',
                            data_level='l3',
                            start_date=datetime(2025, 1, 1),
                            end_date=datetime(2026, 1, 1),
                            version='v001',
                            descriptor=f'hic-ena-h-hf-sp-full-hae-{deg}-1yr'
                        )
                    )
                ]

                initializer = HiCombinedInitializer()
                actual_maps_to_produce = initializer.get_maps_that_should_be_produced(
                    f'hic-ena-h-hf-sp-full-hae-{deg}-1yr')

                self.hi_combined_initializer_mock_query.assert_has_calls([
                    call(instrument='hi', data_level='l3'),
                    call(instrument='hi', data_level='l2')
                ])

                self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    def test_get_maps_that_should_be_produced_no_existing_combined_nsp(self):
        self.maxDiff = 2000
        for deg in ["6deg", "4deg"]:
            with self.subTest(msg=f"'hic-ena-h-hf-sp-full-hae-{deg}-1yr'"):
                self.hi_combined_initializer_mock_query.side_effect = [
                    create_mock_query_results([]),
                    create_mock_query_results([
                        f'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                        f'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-{deg}-1yr_20250101_v001.cdf',

                        f'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                        f'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-{deg}-1yr_20250101_v001.cdf',
                    ])
                ]

                expected_maps_to_produce = [
                    PossibleMapToProduce(
                        input_files={
                            f'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                            f'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-{deg}-1yr_20250101_v001.cdf',

                            f'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-{deg}-1yr_20250101_v001.cdf',
                            f'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-{deg}-1yr_20250101_v001.cdf',
                        },
                        input_metadata=InputMetadata(
                            instrument='hi',
                            data_level='l3',
                            start_date=datetime(2025, 1, 1),
                            end_date=datetime(2026, 1, 1),
                            version='v001',
                            descriptor=f'hic-ena-h-hf-nsp-full-hae-{deg}-1yr'
                        )
                    )
                ]

                initializer = HiCombinedInitializer()
                actual_maps_to_produce = initializer.get_maps_that_should_be_produced(
                    f'hic-ena-h-hf-nsp-full-hae-{deg}-1yr')

                self.hi_combined_initializer_mock_query.assert_has_calls([
                    call(instrument='hi', data_level='l3'),
                    call(instrument='hi', data_level='l2')
                ])

                self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_waits_for_full_set_of_new_inputs(self,
                                                                               mock_read_cdf_parents: Mock):
        self.hi_combined_initializer_mock_query.side_effect = [
            create_mock_query_results([
                *l3_sp_input_maps,
                'imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',

                'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20260101_v002.cdf',
                'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20260101_v002.cdf',
                'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20260101_v002.cdf',
            ]),
            create_mock_query_results([])
        ]

        mock_read_cdf_parents.return_value = {
            *l3_sp_input_maps
        }

        expected_maps_to_produce = []

        initializer = HiCombinedInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced('hic-ena-h-hf-sp-full-hae-6deg-1yr')

        self.hi_combined_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            call(instrument='hi', data_level='l2')
        ])

        mock_read_cdf_parents.assert_called_once_with('imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf')

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    def test_get_maps_that_should_be_produced_filters_based_on_input_descriptor(self):
        self.hi_combined_initializer_mock_query.side_effect = [
            create_mock_query_results([
                *l3_sp_input_maps,
            ]),
            create_mock_query_results([])
        ]

        expected_maps_to_produce = []

        initializer = HiCombinedInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced('hic-ena-h-hf-nsp-full-hae-6deg-1yr')

        self.hi_combined_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            call(instrument='hi', data_level='l2')
        ])

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_increments_versions_correctly(self, mock_read_cdf_parents):
        self.hi_combined_initializer_mock_query.side_effect = [
            create_mock_query_results([
                'imap_hi_l3_hic-ena-h-hf-sp-full-hae-6deg-1yr_20250101_v001.cdf',

                'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v002.cdf',
                *l3_sp_input_maps,
            ]),
            create_mock_query_results([])
        ]

        expected_maps_to_produce = [
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l3_h45-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                    'imap_hi_l3_h90-ena-h-hf-sp-ram-hae-6deg-1yr_20250101_v001.cdf',
                    'imap_hi_l3_h90-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v001.cdf',

                    'imap_hi_l3_h45-ena-h-hf-sp-anti-hae-6deg-1yr_20250101_v002.cdf',

                },
                input_metadata=InputMetadata(
                    instrument='hi',
                    data_level='l3',
                    start_date=datetime(2025, 1, 1),
                    end_date=datetime(2026, 1, 1),
                    version='v002',
                    descriptor='hic-ena-h-hf-sp-full-hae-6deg-1yr'
                )
            )
        ]

        mock_read_cdf_parents.return_value = [
            *l3_sp_input_maps
        ]

        initializer = HiCombinedInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced('hic-ena-h-hf-sp-full-hae-6deg-1yr')

        self.hi_combined_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            call(instrument='hi', data_level='l2')
        ])

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_can_produce_multiple_maps(self, mock_read_cdf_parents):
        self.hi_combined_initializer_mock_query.side_effect = [
            create_mock_query_results([
            ]),
            create_mock_query_results([
                *l2_nsp_input_maps,
                'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v001.cdf',
                'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-6deg-1yr_20260101_v001.cdf',
                'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v001.cdf',
                'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-6deg-1yr_20260101_v001.cdf',
            ]),
        ]

        expected_maps_to_produce = [
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20250101_v001.cdf',
                    'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-6deg-1yr_20250101_v001.cdf',

                    'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-6deg-1yr_20250101_v001.cdf',
                    'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-6deg-1yr_20250101_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument='hi',
                    data_level='l3',
                    start_date=datetime(2025, 1, 1),
                    end_date=datetime(2026, 1, 1),
                    version='v001',
                    descriptor=f'hic-ena-h-hf-nsp-full-hae-6deg-1yr'
                )
            ),
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v001.cdf',
                    'imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-6deg-1yr_20260101_v001.cdf',

                    'imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-6deg-1yr_20260101_v001.cdf',
                    'imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-6deg-1yr_20260101_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument='hi',
                    data_level='l3',
                    start_date=datetime(2026, 1, 1),
                    end_date=datetime(2027, 1, 1),
                    version='v001',
                    descriptor=f'hic-ena-h-hf-nsp-full-hae-6deg-1yr'
                )
            ),
        ]

        initializer = HiCombinedInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced(f'hic-ena-h-hf-nsp-full-hae-6deg-1yr')

        mock_read_cdf_parents.assert_not_called()

        self.hi_combined_initializer_mock_query.assert_has_calls([
            call(instrument='hi', data_level='l3'),
            call(instrument='hi', data_level='l2')
        ])

        self.assertEqual(expected_maps_to_produce, actual_maps_to_produce)
