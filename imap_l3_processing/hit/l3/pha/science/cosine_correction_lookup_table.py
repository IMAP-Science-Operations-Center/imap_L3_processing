import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from imap_l3_processing.hit.l3.pha.pha_event_reader import Detector


class DetectorSide(Enum):
    A = 0
    B = 1


class DetectorRange(Enum):
    R2 = 2
    R3 = 3
    R4 = 4


@dataclass(frozen=True)
class DetectedRange:
    range: DetectorRange
    side: DetectorSide


class CosineCorrectionLookupTable:
    _l1_detector_order = ["L1A0a", "L1A0b", "L1A0c", "L1A1a", "L1A1b", "L1A1c", "L1A2a", "L1A2b", "L1A2c", "L1A3a",
                          "L1A3b", "L1A3c", "L1A4a", "L1A4b", "L1A4c"]
    _l2_detector_order = ["L2A0", "L2A1", "L2A2", "L2A3", "L2A4", "L2A5", "L2A6", "L2A7", "L2A8", "L2A9"]

    def __init__(self, range_2A_file_path: Path, range_3A_file_path: Path, range_4A_file_path: Path,
                 range_2B_file_path: Path, range_3B_file_path: Path, range_4B_file_path: Path):
        self._range2A_corrections = {}
        self._range3A_corrections = {}
        self._range4A_corrections = {}
        self._range2B_corrections = {}
        self._range3B_corrections = {}
        self._range4B_corrections = {}

        for path, lookup_table in [(range_2A_file_path, self._range2A_corrections),
                                   (range_3A_file_path, self._range3A_corrections),
                                   (range_4A_file_path, self._range4A_corrections),
                                   (range_2B_file_path, self._range2B_corrections),
                                   (range_3B_file_path, self._range3B_corrections),
                                   (range_4B_file_path, self._range4B_corrections)]:

            with open(str(path)) as file:
                detected_range_reader = csv.reader(file, delimiter=",")

                for row_num, row in enumerate(detected_range_reader):
                    for col_num, value in enumerate(row):
                        lookup_key = (self._l1_detector_order[col_num][3:], self._l2_detector_order[row_num][3:])
                        lookup_table[lookup_key] = float(value)

    def get_cosine_correction(self, detected_range: DetectedRange, l1_detector: Detector,
                              l2_detector: Detector) -> float:
        corrections_by_range = {
            DetectedRange(DetectorRange.R2, DetectorSide.A): self._range2A_corrections,
            DetectedRange(DetectorRange.R3, DetectorSide.A): self._range3A_corrections,
            DetectedRange(DetectorRange.R4, DetectorSide.A): self._range4A_corrections,
            DetectedRange(DetectorRange.R2, DetectorSide.B): self._range2A_corrections,
            DetectedRange(DetectorRange.R3, DetectorSide.B): self._range3A_corrections,
            DetectedRange(DetectorRange.R4, DetectorSide.B): self._range4A_corrections,
        }
        return corrections_by_range[detected_range][(l1_detector.segment, l2_detector.segment)]
