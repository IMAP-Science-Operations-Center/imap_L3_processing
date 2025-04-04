from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class TimeOfFlightLookup:
    tof_to_nanoseconds_lookup: ndarray

    def convert_to_nanoseconds(self, tof):
        return self.tof_to_nanoseconds_lookup[tof]

    @classmethod
    def from_files(cls, file_path):
        tof_lookup = np.loadtxt(file_path, usecols=1, delimiter=',', skiprows=1)
        return cls(tof_lookup)
