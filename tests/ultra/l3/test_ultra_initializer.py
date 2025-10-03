import unittest
from datetime import datetime
from unittest.mock import patch, call

from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.ultra_initializer import UltraInitializer
from tests.integration.integration_test_helpers import create_mock_query


class TestUltraInitializer(unittest.TestCase):
    def test_is_map_initializer(self):
        self.assertIsInstance(UltraInitializer(), MapInitializer)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    @patch('imap_l3_processing.ultra.l3.ultra_initializer.imap_data_access.query')
    def test_get_maps_that_should_be_produced(self, mock_query, mock_read_cdf_parents):
        sensors = [
            "90",
            "45"
        ]

        for sensor in sensors:
            with self.subTest(sensor=sensor):
                mock_query.side_effect = create_mock_query([
                    'imap_glows_l3e_survival-probability-ul_20100101-repoint00001_v001.cdf',
                    'imap_glows_l3e_survival-probability-ul_20100102-repoint00002_v001.cdf',
                    'imap_glows_l3e_survival-probability-ul_20100103-repoint00003_v001.cdf',

                    'imap_glows_l3e_survival-probability-ul_20100401-repoint00101_v002.cdf',
                    'imap_glows_l3e_survival-probability-ul_20100402-repoint00102_v002.cdf',
                    'imap_glows_l3e_survival-probability-ul_20100403-repoint00103_v002.cdf',

                    'imap_glows_l3e_survival-probability-ul_20100703-repoint00201_v001.cdf',

                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100701_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20101001_v001.cdf',

                    f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100401_v001.cdf'
                ], assert_uses_latest=True)

                mock_read_cdf_parents.side_effect = [
                    [
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100103-repoint00003_v001.cdf"
                    ],
                    [
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100403-repoint00103_v001.cdf'
                    ],
                    [
                        'imap_glows_l3e_survival-probability-ul_20100101-repoint00001_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul_20100102-repoint00002_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul_20100103-repoint00003_v001.cdf',
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100103-repoint00003_v001.cdf",
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'
                    ],
                    [
                        'imap_glows_l3e_survival-probability-lo_20100401-repoint00101_v001.cdf',
                        'imap_glows_l3e_survival-probability-lo_20100402-repoint00102_v001.cdf',
                        'imap_glows_l3e_survival-probability-lo_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'
                    ]
                ]

                initializer = UltraInitializer()
                actual_maps_to_produce = initializer.get_maps_that_should_be_produced(
                    f"u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo")

                mock_read_cdf_parents.assert_has_calls([
                    call(f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
                    call(f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'),

                    call(f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100101_v001.cdf'),
                    call(f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100401_v001.cdf'),
                ])

                expected_possible_map_to_produce = PossibleMapToProduce(
                    input_metadata=InputMetadata(
                        instrument="ultra",
                        data_level="l3",
                        start_date=datetime(2010, 4, 1),
                        end_date=datetime(2010, 4, 1),
                        version="v002",
                        descriptor=f"u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo"
                    ),
                    input_files={
                        'imap_glows_l3e_survival-probability-ul_20100401-repoint00101_v002.cdf',
                        'imap_glows_l3e_survival-probability-ul_20100402-repoint00102_v002.cdf',
                        'imap_glows_l3e_survival-probability-ul_20100403-repoint00103_v002.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                    }
                )

                self.assertEqual([expected_possible_map_to_produce], actual_maps_to_produce)
