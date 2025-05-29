import unittest

from imap_l3_processing.glows.l3bc.science.filter_out_bad_days import filter_l3a_files
from tests.test_helpers import get_test_data_path


class TestFilterL3aFiles(unittest.TestCase):
    def test_filter_l3a_files(self):
        l3a_data = [
            create_l3a_dict("2010-01-05 00:00:00", "2010-01-05 07:28:00"),
            create_l3a_dict("2010-01-06 00:00:00", "2010-01-06 07:28:00"),
            create_l3a_dict("2010-01-10 00:00:00", "2010-01-10 07:28:00"),
            create_l3a_dict("2010-01-11 00:00:00", "2010-01-11 07:28:00"),
            create_l3a_dict("2010-01-15 00:00:00", "2010-01-15 07:28:00"),
            create_l3a_dict("2010-01-16 00:00:00", "2010-01-16 07:28:00"),
            create_l3a_dict("2010-01-19 00:00:00", "2010-01-20 00:00:00"),
            create_l3a_dict("2010-01-30 00:00:00", "2010-01-30 07:28:00"),
            create_l3a_dict("2010-01-30 12:00:00", "2010-01-30 21:00:00"),
            create_l3a_dict("2010-01-30 19:00:00", "2010-01-30 23:59:00")
        ]

        expected_filtered_list = [
            create_l3a_dict("2010-01-05 00:00:00", "2010-01-05 07:28:00"),
            create_l3a_dict("2010-01-06 00:00:00", "2010-01-06 07:28:00"),
            create_l3a_dict("2010-01-16 00:00:00", "2010-01-16 07:28:00"),
            create_l3a_dict("2010-01-19 00:00:00", "2010-01-20 00:00:00"),
            create_l3a_dict("2010-01-30 00:00:00", "2010-01-30 07:28:00"),
            create_l3a_dict("2010-01-30 12:00:00", "2010-01-30 21:00:00"),
        ]

        filtered_list = filter_l3a_files(l3a_data=l3a_data, bad_day_list_path=get_test_data_path(
            "glows") / "imap_glows_bad-days-list_v001.dat", cr=2092)
        self.assertEqual(expected_filtered_list, filtered_list)


def create_l3a_dict(start_date: str, end_date: str) -> dict:
    return {
        'start_time': start_date,
        'end_time': end_date,
    }
