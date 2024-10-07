import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class InstrumentResponseLookupTable:
    energy: np.ndarray
    elevation: np.ndarray
    azimuth: np.ndarray
    d_energy: np.ndarray
    d_elevation: np.ndarray
    d_azimuth: np.ndarray
    response: np.ndarray


class InstrumentResponseLookupTableCollection:
    def __init__(self, directory_path: Path):
        self.files = {}
        for file_path in directory_path.iterdir():
            number = re.match('response_ESA(\d*).dat', file_path.name)[1]
            transposed = np.loadtxt(file_path).T
            self.files[int(number)] = InstrumentResponseLookupTable(*transposed)

    def get_table_for_energy_bin(self, index: int) -> InstrumentResponseLookupTable:
        return self.files[index]
