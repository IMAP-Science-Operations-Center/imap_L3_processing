import unittest
from datetime import datetime
from unittest.mock import patch, call

from imap_l3_processing.hi.l3.hi_l3_initializer import (HiL3Initializer, PossibleMapToProduce, get_parent_file_names,
                                                        logger)
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import get_test_data_path


class TestHiL3Initializer(unittest.TestCase):
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.get_parent_file_names')
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query')
    def test_get_maps_that_can_be_produced(self, mock_query, mock_get_parent_file_names):
        mock_query.side_effect = [
            [
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'},
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'},
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'}
            ],
            [
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                 'repointing': 1},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                 'repointing': 2},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                 'repointing': 3},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                 'repointing': 101},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                 'repointing': 102},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
                 'repointing': 103},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100701-repoint00201_v001.cdf',
                 'repointing': 201},
            ]
        ]

        mock_get_parent_file_names.side_effect = mock_get_parent_file_names

        expected_possible_maps = [
            PossibleMapToProduce(
                input_files={
                    f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 1, 1),
                    end_date=None,
                    version="v001",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
            PossibleMapToProduce(
                input_files={
                    f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 4, 1),
                    end_date=None,
                    version="v001",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
        ]

        actual_possible_maps = HiL3Initializer.get_maps_that_can_be_produced(
            descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo')

        mock_get_parent_file_names.assert_has_calls([
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
        ])

        mock_query.assert_has_calls([
            call(instrument='hi', data_level='l2', descriptor='h45-ena-h-sf-nsp-anti-hae-4deg-3mo', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
        ])

        self.assertEqual(expected_possible_maps, actual_possible_maps)

    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.get_parent_file_names')
    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.query')
    def test_get_maps_that_should_be_produced(self, mock_query, mock_get_parent_file_names):
        mock_query.side_effect = [
            [
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'},
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'},
                {'file_path': f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'}
            ],
            [
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
                 'repointing': 1},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
                 'repointing': 2},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
                 'repointing': 3},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v002.cdf',
                 'repointing': 101},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v002.cdf',
                 'repointing': 102},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v002.cdf',
                 'repointing': 103},
                {'file_path': f'imap_glows_l3e_survival-probability-hi-45_20100701-repoint00201_v001.cdf',
                 'repointing': 201},
            ],
            [
                {'file_path': f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf',
                 'start_date': "20100101", 'version': 'v001'},
                {'file_path': f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf',
                 'start_date': "20100401", 'version': 'v001'},
            ],
        ]

        mock_get_parent_file_names.side_effect = mock_get_parent_file_names

        expected_possible_maps = [
            PossibleMapToProduce(
                input_files={
                    f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v002.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v002.cdf',
                    f'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v002.cdf',
                },
                input_metadata=InputMetadata(
                    instrument="hi",
                    data_level="l3",
                    start_date=datetime(2010, 4, 1),
                    end_date=None,
                    version="v002",
                    descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo',
                )
            ),
        ]

        actual_possible_maps = HiL3Initializer.get_maps_that_should_be_produced(
            descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo')

        mock_get_parent_file_names.assert_has_calls([
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf'),
            call(f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'),
            call(f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf'),
            call(f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf'),
        ])

        mock_query.assert_has_calls([
            call(instrument='hi', data_level='l2', descriptor='h45-ena-h-sf-nsp-anti-hae-4deg-3mo', version="latest"),
            call(instrument='glows', data_level='l3e', descriptor='survival-probability-hi-45', version="latest"),
            call(instrument='hi', data_level='l3', descriptor='h45-ena-h-sf-sp-anti-hae-4deg-3mo', version="latest"),
        ])

        self.assertEqual(expected_possible_maps, actual_possible_maps)

    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.download')
    def test_get_parent_file_names(self, mock_download):
        expected_filepath = 'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'
        mock_download.return_value = get_test_data_path(
            "hi/fake_l2_maps/imap_hi_l2_h45-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf")

        result = get_parent_file_names(expected_filepath)
        expected_result = {
            "imap_hi_l1c_90sensor-pset_20250416_v001.cdf",
            "imap_hi_l1c_90sensor-pset_20250417_v001.cdf",
        }
        mock_download.assert_called_once_with(expected_filepath)

        self.assertEqual(expected_result, result)

    @patch('imap_l3_processing.hi.l3.hi_l3_initializer.imap_data_access.download')
    def test_get_parent_file_names_for_l2_with_no_parents(self, mock_download):
        expected_filepath = 'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf'
        mock_download.return_value = get_test_data_path(
            "hi/fake_l2_maps/hi45-zirnstein-mondel-6months.cdf")

        with self.assertLogs(logger) as log_context:
            result = get_parent_file_names(expected_filepath)
            expected_result = set()
            mock_download.assert_called_once_with(expected_filepath)

            self.assertEqual(expected_result, result)
        self.assertEqual(["INFO:imap_l3_processing.hi.l3.hi_l3_initializer:Parents attribute was not found on the L2."],
                         log_context.output)


def mock_get_parent_file_names(file_name: str) -> set[str]:
    mapping = {
        f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf': {
            f'imap_hi_l1c_45sensor-pset_20100101-repoint00001_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100102-repoint00002_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100103-repoint00003_v001.cdf',
            f'imap_science_0001.tf',
        },
        f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf': {
            f'imap_hi_l1c_45sensor-pset_20100401-repoint00101_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100402-repoint00102_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100403-repoint00103_v001.cdf',
        },
        f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100701_v001.cdf': {
            f'imap_hi_l1c_45sensor-pset_20100401-repoint00201_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100402-repoint00202_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100403-repoint00203_v001.cdf',
        },
        f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100101_v001.cdf': {
            f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100101_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100101-repoint00001_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100102-repoint00002_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100103-repoint00003_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100101-repoint00001_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100102-repoint00002_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100103-repoint00003_v001.cdf',
            f'imap_science_0001.tf',
        },
        f'imap_hi_l3_h45-ena-h-sf-sp-anti-hae-4deg-3mo_20100401_v001.cdf': {
            f'imap_hi_l2_h45-ena-h-sf-nsp-anti-hae-4deg-3mo_20100401_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100401-repoint00101_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100402-repoint00102_v001.cdf',
            f'imap_hi_l1c_45sensor-pset_20100403-repoint00103_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100401-repoint00101_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100402-repoint00102_v001.cdf',
            f'imap_glows_l3e_survival-probability-hi-45_20100403-repoint00103_v001.cdf',
            f'imap_science_0001.tf',
        }
    }
    return mapping[file_name]
