import re
import zipfile
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
    def __init__(self, lookup_tables: dict[int, InstrumentResponseLookupTable]):
        self.lookup_tables = lookup_tables

    def get_table_for_energy_bin(self, index: int) -> InstrumentResponseLookupTable:
        return self.lookup_tables[index]

    @classmethod
    def from_file(cls, zip_path: Path):
        files = {}
        with zipfile.ZipFile(zip_path) as zip_file:
            for file_name in zip_file.namelist():
                if match := re.search(r'response_ESA(\d*).dat', file_name):
                    number = int(match[1])
                    transposed = np.loadtxt(zip_file.open(file_name)).T
                    files[int(number)] = InstrumentResponseLookupTable(*transposed)
        return cls(files)
