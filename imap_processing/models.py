import abc
import ctypes
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np


@dataclass
class InputMetadata:
    instrument: str
    data_level: str
    start_date: datetime
    end_date: datetime
    version: str

    def to_upstream_data_dependency(self, descriptor: str):
        return UpstreamDataDependency(self.instrument, self.data_level, self.start_date, self.end_date, self.version,
                                      descriptor)


@dataclass
class UpstreamDataDependency(InputMetadata):
    descriptor: str


@dataclass
class DataProductVariable:
    name: str
    value: Union[np.ndarray, int, float]
    cdf_data_type: ctypes.c_long = None
    record_varying: bool = True


@dataclass
class DataProduct(metaclass=abc.ABCMeta):
    input_metadata: UpstreamDataDependency

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplemented
