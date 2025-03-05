from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, DetectorRange, DetectorSide


def double_power_law(e_prime, a1, b1, a2, b2, gamma):
    return ((a1 * e_prime ** b1) ** gamma + (a2 * e_prime ** b2) ** gamma) ** (1 / gamma)


@dataclass
class RangeFitLookup:
    range_2A_table: ndarray
    range_3A_table: ndarray
    range_4A_table: ndarray
    range_2B_table: ndarray
    range_3B_table: ndarray
    range_4B_table: ndarray

    @classmethod
    def from_files(cls, range2A_file: str | Path, range3A_file: str | Path, range4A_file: str | Path,
                   range2B_file: str | Path, range3B_file: str | Path, range4B_file: str | Path):
        return cls(np.loadtxt(range2A_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range3A_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range4A_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range2B_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range3B_file, delimiter=',', usecols=range(6)),
                   np.loadtxt(range4B_file, delimiter=',', usecols=range(6))
                   )

    def evaluate_e_prime(self, detected_range: DetectedRange, energy):
        tables = {
            DetectedRange(DetectorRange.R2, DetectorSide.A): self.range_2A_table,
            DetectedRange(DetectorRange.R3, DetectorSide.A): self.range_3A_table,
            DetectedRange(DetectorRange.R4, DetectorSide.A): self.range_4A_table,
            DetectedRange(DetectorRange.R2, DetectorSide.B): self.range_2B_table,
            DetectedRange(DetectorRange.R3, DetectorSide.B): self.range_3B_table,
            DetectedRange(DetectorRange.R4, DetectorSide.B): self.range_4B_table
        }

        table = tables[detected_range]
        charges = table[:, 0]
        a1, b1, a2, b2, gamma = table[:, 1:].T
        return charges, double_power_law(energy, a1, b1, a2, b2, gamma)
