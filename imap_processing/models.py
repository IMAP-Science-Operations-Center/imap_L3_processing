import abc
import ctypes
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np


@dataclass
class UpstreamDataDependency:
    instrument: str
    data_level: str
    descriptor: str
    start_date: datetime
    end_date: datetime
    version: str


@dataclass
class DataProductVariable:
    name: str
    value: Union[np.ndarray, int, float]
    cdf_data_type: ctypes.c_long = None
    record_varying: bool = True


class DataProduct(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplemented