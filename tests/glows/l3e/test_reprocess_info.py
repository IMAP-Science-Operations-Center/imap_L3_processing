import unittest
from pathlib import Path

from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo, ProductForReprocessing
from tests.test_helpers import get_test_data_folder


class TestReprocessInfo(unittest.TestCase):
    def test_parse_from_ancillary(self):
        test_file: Path = get_test_data_folder() / 'glows' / 'glows_reprocessing_ancillary_file.txt'

        reprocess_info = ReprocessInfo.parse_from_ancillary(test_file)

        expected_products_to_reprocess = [
            ProductForReprocessing("survival-probability-lo", ["repoint00245", "repoint00246"], ["cr02310"]),
            ProductForReprocessing("survival-probability-hi-90", [], ["cr02313"]),
            ProductForReprocessing("survival-probability-hi-45", [], ["cr02313", "cr02314"]),
            ProductForReprocessing("survival-probability-ul-sf", ["repoint0246", "repoint0247"], []),
            ProductForReprocessing("survival-probability-ul-hf", ["repoint0243"], [])
        ]

        self.assertEqual(expected_products_to_reprocess, reprocess_info.products_to_reprocess)


if __name__ == '__main__':
    unittest.main()
