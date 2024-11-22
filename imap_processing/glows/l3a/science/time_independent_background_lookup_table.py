from dataclasses import dataclass
from pathlib import Path
from typing import Union, TypeVar

import numpy as np
import scipy.interpolate

T = TypeVar('T', bound='TimeIndependentBackgroundLookupTable')


@dataclass
class TimeIndependentBackgroundLookupTable:
    longitudes: np.ndarray[float]
    latitudes: np.ndarray[float]
    background_values: np.ndarray[float]

    def lookup(self, lat: Union[np.ndarray[float], float], lon: Union[np.ndarray[float], float]) -> Union[
        np.ndarray[float], float]:
        wrapped_lon = np.mod(lon, 360)
        background_extended = np.append(self.background_values, self.background_values[:, :1], axis=1)
        longitudes_extended = np.append(self.longitudes, self.longitudes[0] + 360)
        return scipy.interpolate.interpn((self.latitudes, longitudes_extended), background_extended,
                                         np.column_stack([lat, wrapped_lon]))

    @classmethod
    def from_file(cls: type[T], path: Path) -> T:
        values = np.loadtxt(path)
        numlat, numlon = values.shape
        longitudes = np.linspace(0, 360, numlon, endpoint=False)
        latitudes = np.linspace(90, -90, numlat, endpoint=True)
        return cls(longitudes=longitudes, latitudes=latitudes, background_values=values)
