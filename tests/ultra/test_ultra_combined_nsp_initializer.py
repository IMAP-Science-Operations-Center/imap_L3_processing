import unittest
from datetime import datetime
from unittest.mock import patch, call

from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.ultra_combined_nsp_initializer import UltraCombinedNSPInitializer
from tests.integration.integration_test_helpers import ImapQueryPatcher


class TestUltraCombinedNSPInitializer(unittest.TestCase):
    def setUp(self):
        self.mock_query_patcher = patch(
            'imap_l3_processing.ultra.ultra_combined_nsp_initializer.imap_data_access.query')
        self.mock_query = self.mock_query_patcher.start()

    def tearDown(self):
        self.mock_query_patcher.stop()

    def test_is_map_initializer(self):
        self.assertIsInstance(UltraCombinedNSPInitializer(), MapInitializer)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced(self, mock_read_cdf_parents):
        self.mock_query.side_effect = ImapQueryPatcher([
            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',

            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v002.cdf',

            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20101001_v001.cdf',

            f'imap_ultra_l3_ulc-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_ultra_l3_ulc-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
            'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
        ])

        mock_read_cdf_parents.side_effect = [
            [
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100103-repoint00003_v001.cdf",

            ],
            [
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100103-repoint00003_v001.cdf"
            ],
            [
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100403-repoint00103_v001.cdf'
            ], [
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100401-repoint00101_v002.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100402-repoint00102_v002.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100403-repoint00103_v002.cdf'
            ],
            [
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100103-repoint00003_v001.cdf",
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                f"imap_ultra_l1c_90sensor-spacecraftpset_20100103-repoint00003_v001.cdf",
                f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
                f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
                'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv',
            ],
            [
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv',
            ]
        ]

        initializer = UltraCombinedNSPInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced(f"ulc-ena-h-sf-nsp-full-hae-4deg-3mo")

        self.mock_query.assert_has_calls([
            call(instrument='ultra', data_level='l2'),
            call(instrument="ultra", table='ancillary', descriptor='l2-energy-bin-group-sizes',
                 version='latest'),
            call(instrument='ultra', data_level='l3')
        ])

        mock_read_cdf_parents.assert_has_calls([
            call(f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v002.cdf'),

            call(f'imap_ultra_l3_ulc-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l3_ulc-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'),
        ])

        expected_possible_map_to_produce = PossibleMapToProduce(
            input_metadata=InputMetadata(
                instrument="ultra",
                data_level="l3",
                start_date=datetime(2010, 4, 1),
                end_date=datetime(2010, 7, 1, 7, 30),
                version="v002",
                descriptor=f"ulc-ena-h-sf-nsp-full-hae-4deg-3mo"
            ),
            input_files={
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100401-repoint00101_v002.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100402-repoint00102_v002.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100403-repoint00103_v002.cdf',
                f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v002.cdf',
                f'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
            }
        )
        self.assertEqual([expected_possible_map_to_produce], actual_maps_to_produce)

    def test_initializer_raises_when_given_invalid_map_descriptor(self):
        initializer = UltraCombinedNSPInitializer()
        with self.assertRaises(ValueError) as e:
            initializer.get_maps_that_should_be_produced(f"ulc-invalid")
        self.assertEqual(e.exception.args, ("Invalid map descriptor: ulc-invalid",))
