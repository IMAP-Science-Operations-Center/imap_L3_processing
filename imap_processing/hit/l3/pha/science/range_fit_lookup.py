from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange


def double_power_law(e_prime, a1, b1, a2, b2, gamma):
    return ((a1 * e_prime ** b1) ** gamma + (a2 * e_prime ** b2) ** gamma) ** (1 / gamma)


@dataclass
class RangeFitLookup:
    range_2_table: ndarray
    range_3_table: ndarray
    range_4_table: ndarray

    @classmethod
    def from_files(cls, range2_file: str | Path, range3_file: str | Path, range4_file: str | Path):
        return cls(np.loadtxt(range2_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range3_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range4_file, delimiter=',', usecols=range(6)))

    def evaluate_e_prime(self, range: DetectedRange, energy):
        tables = {DetectedRange.R2: self.range_2_table,
                  DetectedRange.R3: self.range_3_table,
                  DetectedRange.R4: self.range_4_table}
        table = tables[range]
        charges = table[:, 0]
        a1, b1, a2, b2, gamma = table[:, 1:].T
        return charges, double_power_law(energy, a1, b1, a2, b2, gamma)
