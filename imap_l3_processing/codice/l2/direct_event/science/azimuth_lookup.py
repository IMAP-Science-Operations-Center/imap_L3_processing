from dataclasses import dataclass

import numpy as np


@dataclass
class AzimuthLookup:
    azimuth_lookup: dict[int, int]

    def get_azimuth_by_ssd_id(self, ssd_id) -> int:
        return self.azimuth_lookup[ssd_id]

    @classmethod
    def from_files(cls, csv_file):
        with open(csv_file, 'r') as f:
            azimuth_lookup = np.loadtxt(f, delimiter=',', skiprows=1)
            azimuth_dictionary = {ssd_id: azimuth for ssd_id, azimuth in zip(azimuth_lookup[0], azimuth_lookup[1])}
            return cls(azimuth_dictionary)
