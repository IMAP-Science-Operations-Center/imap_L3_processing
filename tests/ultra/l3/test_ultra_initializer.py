import unittest
from datetime import datetime
from unittest.mock import patch, call

from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.ultra_initializer import UltraInitializer, ULTRA_SP_SPICE_KERNELS
from tests.integration.integration_test_helpers import ImapQueryPatcher


class TestUltraInitializer(unittest.TestCase):
    def test_is_map_initializer(self):
        self.assertIsInstance(UltraInitializer(), MapInitializer)

    def setUp(self):
        self.mock_query_patcher = patch('imap_l3_processing.ultra.l3.ultra_initializer.imap_data_access.query')
        self.mock_query = self.mock_query_patcher.start()

    def tearDown(self):
        self.mock_query_patcher.stop()

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced(self, mock_read_cdf_parents):
        sensors = [
            "90",
            "45"
        ]

        for sensor in sensors:
            with self.subTest(sensor=sensor):
                self.mock_query.side_effect = ImapQueryPatcher([
                    'imap_glows_l3e_survival-probability-ul-sf_20100101-repoint00001_v001.cdf',
                    'imap_glows_l3e_survival-probability-ul-sf_20100102-repoint00002_v001.cdf',
                    'imap_glows_l3e_survival-probability-ul-sf_20100103-repoint00003_v001.cdf',

                    'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v002.cdf',
                    'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v002.cdf',
                    'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v002.cdf',
                    'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v001.cdf',

                    'imap_glows_l3e_survival-probability-ul-sf_20100703-repoint00201_v001.cdf',

                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v000.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100701_v001.cdf',
                    f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20101001_v001.cdf',

                    f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_ultra_l3_u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo_20100401_v001.cdf',
                    'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
                ])

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
                        'imap_glows_l3e_survival-probability-ul-sf_20100101-repoint00001_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100102-repoint00002_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100103-repoint00003_v001.cdf',
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                        f"imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100103-repoint00003_v001.cdf",
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
                        'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv',
                    ],
                    [
                        'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v001.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                        'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv',
                    ]
                ]

                initializer = UltraInitializer()
                actual_maps_to_produce = initializer.get_maps_that_should_be_produced(
                    f"u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo")

                self.mock_query.assert_has_calls([
                    call(instrument='glows', data_level='l3e', descriptor='survival-probability-ul-sf',
                         version="latest"),
                    call(instrument='ultra', data_level='l2'),
                    call(instrument='ultra', data_level='l3'),
                    call(instrument="ultra", table='ancillary', descriptor='l2-energy-bin-group-sizes',
                         version='latest')
                ])

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
                        end_date=datetime(2010, 7, 1, 7, 30),
                        version="v002",
                        descriptor=f"u{sensor}-ena-h-sf-sp-full-hae-4deg-3mo"
                    ),
                    input_files={
                        'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v002.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v002.cdf',
                        'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v002.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                        f'imap_ultra_l1c_{sensor}sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                        f'imap_ultra_l2_u{sensor}-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                        f'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
                    }
                )
                self.assertEqual([expected_possible_map_to_produce], actual_maps_to_produce)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    def test_get_maps_that_should_be_produced_lists_combined_maps(self, mock_read_cdf_parents):
        self.mock_query.side_effect = ImapQueryPatcher([
            'imap_glows_l3e_survival-probability-ul-sf_20100101-repoint00001_v001.cdf',
            'imap_glows_l3e_survival-probability-ul-sf_20100102-repoint00002_v001.cdf',
            'imap_glows_l3e_survival-probability-ul-sf_20100103-repoint00003_v001.cdf',

            'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v002.cdf',
            'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v002.cdf',
            'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v002.cdf',
            'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v001.cdf',

            'imap_glows_l3e_survival-probability-ul-sf_20100703-repoint00201_v001.cdf',

            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v000.cdf',
            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100701_v001.cdf',
            f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20101001_v001.cdf',

            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v000.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100701_v001.cdf',
            f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20101001_v001.cdf',

            f'imap_ultra_l3_ulc-ena-h-sf-sp-full-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_ultra_l3_ulc-ena-h-sf-sp-full-hae-4deg-3mo_20100401_v001.cdf',
            'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
        ])

        mock_read_cdf_parents.side_effect = [
            [
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100101-repoint00001_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100102-repoint00002_v001.cdf",
                f"imap_ultra_l1c_45sensor-spacecraftpset_20100103-repoint00003_v001.cdf"
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
            ],
            [
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100403-repoint00103_v001.cdf'
            ],
            [
                'imap_glows_l3e_survival-probability-ul-sf_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100103-repoint00003_v001.cdf',
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
                'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v001.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v001.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v001.cdf',
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

        initializer = UltraInitializer()
        actual_maps_to_produce = initializer.get_maps_that_should_be_produced(
            f"ulc-ena-h-sf-sp-full-hae-4deg-3mo")

        self.mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-ul-sf', version="latest"),
            call(instrument='ultra', data_level='l2'),
            call(instrument='ultra', data_level='l3'),
            call(instrument="ultra", table='ancillary', descriptor='l2-energy-bin-group-sizes',
                 version='latest')
        ])

        mock_read_cdf_parents.assert_has_calls([
            call(f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf'),

            call(f'imap_ultra_l3_ulc-ena-h-sf-sp-full-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_ultra_l3_ulc-ena-h-sf-sp-full-hae-4deg-3mo_20100401_v001.cdf'),
        ])

        expected_possible_map_to_produce = PossibleMapToProduce(
            input_metadata=InputMetadata(
                instrument="ultra",
                data_level="l3",
                start_date=datetime(2010, 4, 1),
                end_date=datetime(2010, 7, 1, 7, 30),
                version="v002",
                descriptor=f"ulc-ena-h-sf-sp-full-hae-4deg-3mo"
            ),
            input_files={
                'imap_glows_l3e_survival-probability-ul-sf_20100401-repoint00101_v002.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100402-repoint00102_v002.cdf',
                'imap_glows_l3e_survival-probability-ul-sf_20100403-repoint00103_v002.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_45sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100401-repoint00101_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100402-repoint00102_v001.cdf',
                f'imap_ultra_l1c_90sensor-spacecraftpset_20100403-repoint00103_v001.cdf',
                f'imap_ultra_l2_u45-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                f'imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20100401_v001.cdf',
                f'imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv'
            }
        )
        self.assertEqual([expected_possible_map_to_produce], actual_maps_to_produce)

    @patch('imap_l3_processing.ultra.l3.ultra_initializer.furnish_spice_metakernel')
    def test_furnish_spice_dependencies(self, mock_furnish_metakernel):
        start_date = datetime(2025, 4, 15)
        end_date = datetime(2025, 7, 15)

        input_metadata = InputMetadata(
            instrument="ultra",
            data_level="l2",
            start_date=start_date,
            end_date=end_date,
            version="v000",
            descriptor="u90-ena-h-sf-sp-full-hae-4deg-3mo",
        )
        map_to_produce = PossibleMapToProduce(set(), input_metadata)

        ultra_initializer = UltraInitializer()
        ultra_initializer.furnish_spice_dependencies(map_to_produce)

        mock_furnish_metakernel.assert_called_once_with(start_date=start_date, end_date=end_date,
                                                        kernel_types=ULTRA_SP_SPICE_KERNELS)