import abc
import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Union

import numpy as np

from imap_processing.data_utils import rebin


@dataclass
class InputMetadata:
    instrument: str
    data_level: str
    start_date: datetime
    end_date: datetime
    version: str
    descriptor: str = ""

    def to_upstream_data_dependency(self, descriptor: str):
        return UpstreamDataDependency(self.instrument, self.data_level, self.start_date, self.end_date, self.version,
                                      descriptor)


@dataclass
class UpstreamDataDependency(InputMetadata):
    descriptor: str

    @property
    def logical_source(self):
        return f"imap_{self.instrument}_{self.data_level}_{self.descriptor}"


@dataclass
class DataProductVariable:
    name: str
    value: Union[np.ndarray, int, float, list[str]]
    cdf_data_type: ctypes.c_long = None
    record_varying: bool = True


@dataclass
class DataProduct(metaclass=abc.ABCMeta):
    input_metadata: UpstreamDataDependency

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplemented


@dataclass
class MagL1dData:
    epoch: np.ndarray[float]
    mag_data: np.ndarray[float]

    def rebin_to(self, epoch: np.ndarray[float], epoch_delta: np.ndarray[timedelta]) -> np.ndarray[float]:
        return rebin(self.epoch, self.mag_data, epoch, epoch_delta)
