import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self


@dataclass
class Gain:
    a: float
    b: float


class DetectorGain(Enum):
    LOW = True
    HIGH = False


class GainLookupTable(dict):
    @classmethod
    def from_file(cls, high_gain_file_path: Path, low_gain_file_path: Path) -> Self:
        lookup_table = cls()
        high_gain_lookup = {}
        low_gain_lookup = {}

        with open(str(high_gain_file_path)) as gain_file:
            gain_reader = csv.reader(gain_file, delimiter=" ")
            for detector_address, a, b in gain_reader:
                high_gain_lookup[int(detector_address)] = Gain(a=float(a), b=float(b))

        with open(str(low_gain_file_path)) as gain_file:
            gain_reader = csv.reader(gain_file, delimiter=" ")
            for detector_address, a, b in gain_reader:
                low_gain_lookup[int(detector_address)] = Gain(a=float(a), b=float(b))

        lookup_table[DetectorGain.HIGH] = high_gain_lookup
        lookup_table[DetectorGain.LOW] = low_gain_lookup

        return lookup_table
