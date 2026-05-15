from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np


@dataclass
class InflowVector:
    speed_km_per_s: float
    longitude_deg_eclipj2000: float
    latitude_deg_eclipj2000: float

    @classmethod
    def from_file(cls, path: PathLike) -> InflowVector:
        path = Path(path)

        loaded_vector = np.squeeze(np.loadtxt(path, dtype=float))

        assert loaded_vector.shape == (3,), f"Failed to parse Inflow Vector from {path.name}"

        return cls(
            speed_km_per_s=(loaded_vector[0]),
            longitude_deg_eclipj2000=loaded_vector[1],
            latitude_deg_eclipj2000=loaded_vector[2]
        )
