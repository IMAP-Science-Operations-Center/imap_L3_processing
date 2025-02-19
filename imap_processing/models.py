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

    def rebin_to(self, epoch: np.ndarray[float], epoch_delta: np.ndarray[float]) -> np.ndarray[float]:
        vector_sums = np.zeros(shape=(epoch.shape[0], 3), dtype=float)
        vector_counts = np.zeros_like(epoch, dtype=float)

        mag_data_iter = ((time, vec) for time, vec in zip(self.epoch, self.mag_data))
        current_epoch, current_vec = next(mag_data_iter)
        for i, (time, delta) in enumerate(zip(epoch, epoch_delta)):
            start_time = time - delta
            end_time = time + delta

            while current_epoch < start_time:
                current_epoch, current_vec = next(mag_data_iter)

            while current_epoch is not None and start_time <= current_epoch < end_time:
                vector_sums[i] += current_vec
                vector_counts[i] += 1
                current_epoch, current_vec = next(mag_data_iter, (None, None))

        vector_counts = np.reshape(vector_counts, (-1, 1))
        return np.divide(vector_sums, vector_counts, out=np.full_like(vector_sums, fill_value=np.nan),
                         where=vector_counts != 0)
