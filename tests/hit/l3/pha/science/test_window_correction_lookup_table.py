import unittest
from pathlib import Path

from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectorRange
from imap_l3_processing.hit.l3.pha.science.window_correction_lookup_table import (
    WindowCorrectionLookupTable,
)
from tests.test_helpers import get_test_data_path, with_tempdir


class TestWindowCorrectionLookupTable(unittest.TestCase):
    def test_returns_wincorr_entry_over_256_for_range_and_detector_address(self) -> None:
        table = WindowCorrectionLookupTable.from_file(
            get_test_data_path("hit/pha_events/imap_hit_wincorr_20250901_v000.csv")
        )

        cases = [
            (DetectorRange.R2, 16, 391),
            (DetectorRange.R2, 19, 390),
            (DetectorRange.R2, 32, 391),
            (DetectorRange.R2, 9, 256),
            (DetectorRange.R3, 18, 420),
            (DetectorRange.R3, 35, 419),
            (DetectorRange.R4, 30, 419),
            (DetectorRange.R4, 63, 256),
        ]
        for detector_range, address, wincorr_value in cases:
            with self.subTest(range=detector_range, address=address):
                self.assertEqual(
                    wincorr_value / 256,
                    table.get_correction_factor(detector_range, address),
                )

    @with_tempdir
    def test_rejects_wincorr_file_without_two_rows_of_64_values(self, temp_dir: Path) -> None:
        row_of_64 = ",".join(["256"] * 64)
        shapes = {
            "one row of 64 values": [row_of_64],
            "three rows of 64 values": [row_of_64, row_of_64, row_of_64],
        }
        for shape, rows in shapes.items():
            path = temp_dir / f"{shape}.csv"
            path.write_text("# provenance\n" + "\n".join(rows) + "\n")
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError) as context:
                    WindowCorrectionLookupTable.from_file(path)
                self.assertIn(str(path), str(context.exception))

        jagged_path = temp_dir / "two rows where the second has 63 values.csv"
        jagged_path.write_text("# provenance\n" + row_of_64 + "\n" + ",".join(["256"] * 63) + "\n")
        with self.assertRaises(ValueError):
            WindowCorrectionLookupTable.from_file(jagged_path)
