import unittest
from datetime import datetime
from typing import Optional

import numpy as np
from imap_processing.spice.repoint import set_global_repoint_table_paths

from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range, get_best_ancillary
from tests.test_helpers import get_test_data_path


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repoint_file_path = get_test_data_path("fake_1_day_repointing_file.csv")
        set_global_repoint_table_paths([repoint_file_path])

    def test_determine_crs_to_process_based_on_ancillary_files(self):
        start_date = datetime(2009, 12, 31)
        end_date = datetime(2010, 1, 2)

        case_1 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100102 00:00:00"}
        ]

        case_2 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20100104", "end_date": None,
             "ingestion_date": "20100102 00:00:00"}
        ]

        case_3 = [
            {"file_path": "some/server/path/" + "older_ancillary.dat", "start_date": "20100101", "end_date": None,
             "ingestion_date": "20100101 00:00:00"},
            {"file_path": "some/server/path/" + "newer_ancillary.dat", "start_date": "20091205", "end_date": "20091206",
             "ingestion_date": "20100102 00:00:00"}
        ]

        test_cases = [
            ("picks latest ingestion date", case_1, "newer_ancillary.dat"),
            ("ignores ancillary with start date after cr", case_2, "older_ancillary.dat"),
            ("ignores ancillary with end date before cr", case_3, "older_ancillary.dat")
        ]

        for name, available_ancillaries, expected_best_ancillary_file_name in test_cases:
            with self.subTest(name):
                actual_ancillary_name = get_best_ancillary(start_date, end_date, available_ancillaries)

                self.assertEqual(expected_best_ancillary_file_name, actual_ancillary_name)

    def test_get_pointing_date_range(self):
        repointing_number = 13
        actual_start, actual_end = get_pointing_date_range(repointing_number)
        expected_start = np.datetime64(
            datetime(year=2000, month=1, day=14, hour=12, minute=13, second=55, microsecond=816000).isoformat())
        expected_end = np.datetime64(
            datetime(year=2000, month=1, day=15, hour=11, minute=58, second=55, microsecond=816000).isoformat())

        np.testing.assert_array_equal(actual_start, expected_start)
        np.testing.assert_array_equal(actual_end, expected_end)

    def test_get_repoint_date_range_handles_no_pointing(self):
        repointing_number = 5998
        with self.assertRaises(ValueError) as err:
            _, _ = get_pointing_date_range(repointing_number)
        self.assertEqual(str(err.exception), f"No pointing found for pointing: 5998")


def create_imap_data_access_json(file_path: str, data_level: str, start_date: str,
                                 descriptor: str = "hist", version: str = "v001",
                                 repointing: Optional[int] = None) -> dict:
    return {'file_path': file_path, 'instrument': 'glows', 'data_level': data_level, 'descriptor': descriptor,
            'start_date': start_date, 'repointing': repointing, 'version': version, 'extension': 'pkts',
            'ingestion_date': '2024-10-11 15:28:32'}


def create_l3a_path_by_date(file_date: str, repointing: int) -> str:
    return f'imap/glows/l3a/2000/01/imap_glows_l3a_hist_{file_date}-repoint{str(repointing).zfill(5)}_v001.pkts'
