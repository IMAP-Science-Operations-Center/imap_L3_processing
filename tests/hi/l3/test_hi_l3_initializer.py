import unittest
from datetime import datetime
from typing import Callable
from unittest.mock import patch, call, Mock, sentinel

from imap_l3_processing.hi.l3.hi_l3_initializer import HiL3Initializer, HI_SP_SPICE_KERNELS
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import create_mock_query_results


class TestHiL3Initializer(unittest.TestCase):
    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query')
    def test_get_maps_that_can_be_produced(self, mock_query, mock_read_cdf_parents):
        mock_query.side_effect = [
            create_mock_query_results("glows", [
                'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20100701-repoint00201_v001.cdf',
            ]),
            create_mock_query_results("glows", []),
            create_mock_query_results("hi", [
                'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'
            ]),
            create_mock_query_results("hi", [])
        ]

        mock_read_cdf_parents.side_effect = self.create_fake_read_cdf_parents("45")

        expected_possible_maps = [
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 1, 1),
                    end_date=datetime(2010, 1, 1),
                    version="v001",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 4, 1),
                    end_date=datetime(2010, 4, 1),
                    version="v001",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-45_20100701-repoint00201_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 7, 1),
                    end_date=datetime(2010, 7, 1),
                    version="v001",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
        ]

        initializer = HiL3Initializer()

        mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2', version="latest"),
            call(instrument='hi', data_level='l3', version="latest"),
        ])

        actual_possible_maps = initializer.get_maps_that_can_be_produced('h45-ena-h-sf-sp-anti-hae-4deg-3mo')

        mock_read_cdf_parents.assert_has_calls([
            call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
            call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
            call('imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'),
        ], any_order=True)

        self.assertEqual(expected_possible_maps, actual_possible_maps)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query')
    def test_get_maps_that_can_be_produced_full_spin_descriptor(self, mock_query, mock_read_cdf_parents):
        mock_query.side_effect = [
            create_mock_query_results("glows", []),
            create_mock_query_results("glows", [
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100104-repoint00004_v001.cdf',
            ]),
            create_mock_query_results("hi", [
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-3mo_20100701_v001.cdf',
            ]),
            create_mock_query_results("hi", [])
        ]

        mock_read_cdf_parents.side_effect = [
            [
                f'imap_hi_l1c_90sensor-pset_20100101-repoint00001_v001.cdf',
                f'imap_hi_l1c_90sensor-pset_20100102-repoint00002_v001.cdf',
            ],
            [
                f'imap_hi_l1c_90sensor-pset_20100103-repoint00003_v001.cdf',
            ]
        ]

        initializer = HiL3Initializer()
        actual_possible_maps = initializer.get_maps_that_can_be_produced('h90-ena-h-sf-sp-full-hae-4deg-3mo')

        mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2', version="latest"),
            call(instrument='hi', data_level='l3', version="latest"),
        ])

        mock_read_cdf_parents.assert_has_calls([
            call('imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call('imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-3mo_20100101_v001.cdf'),
        ])

        expected_possible_map_to_produce = PossibleMapToProduce(
            input_files={
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-3mo_20100101_v001.cdf',
            },
            input_metadata=InputMetadata(
                instrument="hi",
                data_level="l3",
                start_date=datetime(2010, 1, 1),
                end_date=datetime(2010, 1, 1),
                version="v001",
                descriptor='h90-ena-h-sf-sp-full-hae-4deg-3mo',
            )
        )

        self.assertEqual([expected_possible_map_to_produce], actual_possible_maps)

    @patch('imap_l3_processing.maps.map_initializer.read_cdf_parents')
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query')
    def test_get_maps_that_should_be_produced(self, mock_query, mock_read_cdf_parents):
        mock_query.side_effect = [
            create_mock_query_results("glows", []),
            create_mock_query_results("glows", [
                'imap_glows_l3e_survival-probability-hi-90_20100101-repoint00001_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100102-repoint00002_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100103-repoint00003_v001.cdf',

                'imap_glows_l3e_survival-probability-hi-90_20100401-repoint00101_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100402-repoint00102_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20100403-repoint00103_v002.cdf',

                'imap_glows_l3e_survival-probability-hi-90_20100701-repoint00201_v001.cdf',
            ]),
            create_mock_query_results("hi", [
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'

            ]),
            create_mock_query_results("hi", [
                'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf',
                'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf',
            ]),
        ]

        mock_read_cdf_parents.side_effect = self.create_fake_read_cdf_parents("90")

        expected_possible_maps = [
            PossibleMapToProduce(
                input_files={
                    f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-90_20100401-repoint00101_v002.cdf',
                    f'imap_glows_l3e_survival-probability-hi-90_20100402-repoint00102_v002.cdf',
                    f'imap_glows_l3e_survival-probability-hi-90_20100403-repoint00103_v002.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 4, 1),
                    end_date=datetime(2010, 4, 1),
                    version="v002",
                    descriptor='h90-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
            PossibleMapToProduce(
                input_files={
                    'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf',
                    'imap_glows_l3e_survival-probability-hi-90_20100701-repoint00201_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 7, 1),
                    end_date=datetime(2010, 7, 1),
                    version="v001",
                    descriptor='h90-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
        ]

        initializer = HiL3Initializer()

        mock_query.assert_has_calls([
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-90', version="latest"),
            call(instrument='hi', data_level='l2', version="latest"),
            call(instrument='hi', data_level='l3', version="latest"),
        ])

        actual_possible_maps = initializer.get_maps_that_should_be_produced('h90-ena-h-sf-sp-anti-hae-4deg-3mo')

        mock_read_cdf_parents.assert_has_calls([
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
            call(f'imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf'),

            call(f'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l3_h90-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf'),
        ])

        self.assertEqual(expected_possible_maps, actual_possible_maps)

    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query', return_value=[])
    def test_get_dependencies(self, _):
        cases = [
            ("h90-ena-h-sf-sp-anti-hae-4deg-3mo", ["h90-ena-h-sf-nsp-anti-hae-4deg-3mo"]),
            ("h45-ena-h-sf-sp-ram-hae-4deg-3mo", ["h45-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
            ("h45-ena-h-sf-sp-full-hae-4deg-3mo",
             ["h45-ena-h-sf-nsp-anti-hae-4deg-3mo", "h45-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
            ("h90-ena-h-sf-sp-full-hae-4deg-3mo",
             ["h90-ena-h-sf-nsp-anti-hae-4deg-3mo", "h90-ena-h-sf-nsp-ram-hae-4deg-3mo"]),
        ]

        for descriptor, dependencies in cases:
            with self.subTest(descriptor):
                initializer = HiL3Initializer()
                actual_dependencies = initializer._get_l2_dependencies(descriptor)
                self.assertEqual(dependencies, actual_dependencies)

    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.furnish_spice_metakernel')
    def test_furnish_spice_dependencies(self, mock_furnish_metakernel):
        start_date = datetime(2025, 4, 15)
        end_date = datetime(2025, 7, 15)

        input_metadata = InputMetadata(
            instrument="hi",
            data_level="l2",
            start_date=start_date,
            end_date=end_date,
            version="v000",
            descriptor="h90-ena-h-sf-nsp-anti-hae-4deg-3mo",
        )
        map_to_produce = PossibleMapToProduce(set(), input_metadata)

        hi_initializer = HiL3Initializer()
        hi_initializer.furnish_spice_dependencies(map_to_produce)

        mock_furnish_metakernel.assert_called_once_with(start_date=start_date, end_date=end_date,
                                                        kernel_types=HI_SP_SPICE_KERNELS)

    @patch("imap_l3_processing.maps.map_initializer.ScienceInput")
    @patch("imap_l3_processing.maps.map_initializer.ProcessingInputCollection")
    def test_possible_maps_to_produce_constructs_processing_input_collection(self, mock_collection, mock_science_input):
        files = {
            "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v000.cdf",
            "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100102_v000.cdf",
            "imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-3mo_20100103_v000.cdf",
        }

        mock_science_input.side_effect = [sentinel.file1, sentinel.file2, sentinel.file3]

        processing_input = PossibleMapToProduce(input_files=files, input_metadata=Mock()).processing_input_collection

        mock_collection.assert_called_once_with(sentinel.file1, sentinel.file2, sentinel.file3)

        self.assertEqual(mock_collection.return_value, processing_input)

    def create_fake_read_cdf_parents(self, sensor: str) -> Callable[[str], set[str]]:
        def fake_read_cdf_parents(file_name: str) -> set[str]:
            mapping = {
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100101-repoint00001_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100102-repoint00002_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100103-repoint00003_v001.cdf',
                    f'imap_science_0001.tf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00102_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00103_v001.cdf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00201_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00202_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00203_v001.cdf',
                },
                f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20101001_v001.cdf': {
                    f'imap_hi_l1c_{sensor}sensor-pset_20101001-repoint00301_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20101002-repoint00302_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20101003-repoint00303_v001.cdf',
                },
                f'imap_hi_l3_h{sensor}-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf': {
                    f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100101-repoint00001_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100102-repoint00002_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100103-repoint00003_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100101-repoint00001_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100102-repoint00002_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100103-repoint00003_v001.cdf',
                    f'imap_science_0001.tf',
                },
                f'imap_hi_l3_h{sensor}-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf': {
                    f'imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100401-repoint00101_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100402-repoint00102_v001.cdf',
                    f'imap_hi_l1c_{sensor}sensor-pset_20100403-repoint00103_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100401-repoint00101_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100402-repoint00102_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-{sensor}_20100403-repoint00103_v001.cdf',
                    f'imap_science_0001.tf',
                },
            }
            return mapping[file_name]

        return fake_read_cdf_parents
