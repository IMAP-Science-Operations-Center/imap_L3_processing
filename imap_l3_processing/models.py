import abc
import ctypes
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union, Optional

import numpy as np

from imap_l3_processing.data_utils import rebin


@dataclass
class InputMetadata:
    instrument: str
    data_level: str
    start_date: datetime
    end_date: datetime
    version: str
    descriptor: str = ""
    repointing: Optional[int] = None

    @property
    def logical_source(self):
        return f"imap_{self.instrument}_{self.data_level}_{self.descriptor}"

    def to_upstream_data_dependency(self, descriptor: str):
        return UpstreamDataDependency(self.instrument, self.data_level, self.start_date, self.end_date, self.version,
                                      descriptor, self.repointing)


@dataclass
class UpstreamDataDependency(InputMetadata):
    pass


@dataclass
class DataProductVariable:
    name: str
    value: Union[np.ndarray, int, float, list[str]]
    cdf_data_type: ctypes.c_long = None
    record_varying: bool = None


@dataclass
class DataProduct(metaclass=abc.ABCMeta):
    input_metadata: InputMetadata

    parent_file_names: list[str] = field(default_factory=list, kw_only=True)

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplemented


@dataclass
class MagL1dData:
    epoch: np.ndarray
    mag_data: np.ndarray

    def rebin_to(self, epoch: np.ndarray[float], epoch_delta: np.ndarray[float]) -> np.ndarray[float]:
        return rebin(self.epoch, self.mag_data, epoch, epoch_delta)


@dataclass
class MapData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    energy_label: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    exposure_factor: np.ndarray
    obs_date: np.ndarray
    obs_date_range: np.ndarray
    solid_angle: np.ndarray


@dataclass
class HealPixCoords:
    pixel_index: np.ndarray
    pixel_index_label: np.ndarray


@dataclass
class RectangularCoords:
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray


@dataclass
class IntensityMapData(MapData):
    ena_intensity: np.ndarray
    ena_intensity_stat_unc: np.ndarray
    ena_intensity_sys_err: np.ndarray


@dataclass
class SpectralIndexMapData(MapData):
    ena_spectral_index: np.ndarray
    ena_spectral_index_stat_unc: np.ndarray
