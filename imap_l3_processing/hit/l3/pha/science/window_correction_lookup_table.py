from pathlib import Path

import numpy as np

from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectorRange

WINCORR_SCALE = 256
DETECTOR_ADDRESS_COUNT = 64
WINCORR_ROW_COUNT = 2


class WindowCorrectionLookupTable:
    _range_2_factors: np.ndarray
    _range_3_and_4_factors: np.ndarray

    def __init__(self, range_2_wincorr: np.ndarray, range_3_and_4_wincorr: np.ndarray) -> None:
        self._range_2_factors = range_2_wincorr / WINCORR_SCALE
        self._range_3_and_4_factors = range_3_and_4_wincorr / WINCORR_SCALE

    @classmethod
    def from_file(cls, path: Path) -> "WindowCorrectionLookupTable":
        wincorr = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        if wincorr.shape != (WINCORR_ROW_COUNT, DETECTOR_ADDRESS_COUNT):
            raise ValueError(
                f"Expected {WINCORR_ROW_COUNT} rows of {DETECTOR_ADDRESS_COUNT} "
                f"WINCORR values in {path}, got shape {wincorr.shape}"
            )
        return cls(wincorr[0], wincorr[1])

    def get_correction_factor(self, detector_range: DetectorRange, detector_address: int) -> float:
        factors_by_range = {
            DetectorRange.R2: self._range_2_factors,
            DetectorRange.R3: self._range_3_and_4_factors,
            DetectorRange.R4: self._range_3_and_4_factors,
        }
        return float(factors_by_range[detector_range][detector_address])
