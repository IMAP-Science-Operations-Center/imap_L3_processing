import unittest
from pathlib import Path

from imap_l3_processing.glows.descriptors import GLOWS_L3E_HI_45_DESCRIPTOR, GLOWS_L3E_HI_90_DESCRIPTOR, \
    GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3E_ULTRA_HF_DESCRIPTOR, GLOWS_L3B_DESCRIPTOR, \
    GLOWS_L3C_DESCRIPTOR, GLOWS_L3D_DESCRIPTOR
from imap_l3_processing.glows.l3e.reprocess_info import ReprocessInfo, ProductToReprocess
from tests.test_helpers import get_test_data_folder


class TestReprocessInfo(unittest.TestCase):
    def test_parse_from_ancillary(self):
        test_file: Path = get_test_data_folder() / 'glows' / 'glows_reprocessing_ancillary_file.txt'

        reprocess_info = ReprocessInfo.parse_from_ancillary(test_file)

        expected_products_to_reprocess = [
            ProductToReprocess("survival-probability-lo", ["repoint00245", "repoint00246"], ["cr02310"]),
            ProductToReprocess("survival-probability-hi-90", [], ["cr02313"]),
            ProductToReprocess("survival-probability-hi-45", [], ["cr02313", "cr02314"]),
            ProductToReprocess("survival-probability-ul-sf", ["repoint0246", "repoint0247"], []),
            ProductToReprocess("survival-probability-ul-hf", ["repoint0243"], [])
        ]

        self.assertEqual(expected_products_to_reprocess, reprocess_info.products_to_reprocess)

    def test_should_reprocess_l3d_if_l3e_products_are_specified(self):
        l3b_product = ProductToReprocess(GLOWS_L3B_DESCRIPTOR, [], [])

        cases = [
            (f"should reprocess {GLOWS_L3E_HI_45_DESCRIPTOR}", [ProductToReprocess(GLOWS_L3E_HI_45_DESCRIPTOR, [], []), l3b_product], True),
            (f"should reprocess {GLOWS_L3E_HI_90_DESCRIPTOR}", [ProductToReprocess(GLOWS_L3E_HI_90_DESCRIPTOR, [], []), l3b_product], True),
            (f"should reprocess {GLOWS_L3E_LO_DESCRIPTOR}", [ProductToReprocess(GLOWS_L3E_LO_DESCRIPTOR, [], []), l3b_product], True),
            (f"should reprocess {GLOWS_L3E_ULTRA_SF_DESCRIPTOR}", [ProductToReprocess(GLOWS_L3E_ULTRA_SF_DESCRIPTOR, [], []), l3b_product], True),
            (f"should reprocess {GLOWS_L3E_ULTRA_HF_DESCRIPTOR}", [ProductToReprocess(GLOWS_L3E_ULTRA_HF_DESCRIPTOR, [], []), l3b_product], True),
            ("should not reprocess l3b alone", [l3b_product], False),
            ("should not reprocess l3c alone", [ProductToReprocess(GLOWS_L3C_DESCRIPTOR, [], [])], False),
            ("should not reprocess l3d alone", [ProductToReprocess(GLOWS_L3D_DESCRIPTOR, [], [])], False),
        ]

        for case, products_to_reprocess, expected_result in cases:
            reprocess_info = ReprocessInfo(products_to_reprocess)

            self.assertEqual(expected_result, reprocess_info.should_reprocess_l3d())





if __name__ == '__main__':
    unittest.main()
