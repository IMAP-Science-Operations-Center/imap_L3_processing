import abc
import ctypes
import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, TypeVar, Generic

import numpy as np
from imap_data_access.file_validation import Version

from imap_l3_processing.data_utils import rebin


@dataclass
class VersionMap:
    mapping: dict[str, Version]
    fallback: Version | None = None

    def lookup(self, descriptor: str) -> Version:
        result = self.mapping.get(descriptor, self.fallback)
        if result is None:
            raise KeyError(f"No version found for descriptor {descriptor}")
        return result


@dataclass
class InputMetadata:
    instrument: str
    data_level: str
    start_date: datetime
    end_date: Optional[datetime]
    version: VersionMap
    descriptor: str = ""
    repointing: Optional[int] = None

    @property
    def logical_source(self):
        return f"imap_{self.instrument}_{self.data_level}_{self.descriptor}"


@dataclass
class DataProductVariable:
    name: str
    value: Union[np.ndarray, int, float, list[str], list[int]]
    cdf_data_type: ctypes.c_long = None
    record_varying: bool = None


D = TypeVar("D")

@dataclass
class DataProduct(abc.ABC, Generic[D]):
    input_metadata: InputMetadata
    parent_file_names: list[str] = field(default_factory=list, kw_only=True)
    global_metadata_attrs: dict[str, str] = field(default_factory=dict, kw_only=True)

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplemented

    def add_paths_to_parents(self, paths: list[Path]):
        self.add_filenames_to_parents([path.name for path in paths])

    def add_filenames_to_parents(self, filenames):
        self.parent_file_names.extend(filename for filename in filenames if filename not in self.parent_file_names)


@dataclass
class MagData:
    epoch: np.ndarray
    mag_data: np.ndarray

    def rebin_to(self, epoch: np.ndarray[float], epoch_delta: np.ndarray[float]) -> np.ndarray[float]:
        return rebin(self.epoch, self.mag_data, epoch, epoch_delta)


class Instrument(enum.Enum):
    IMAP_HI = "hi"
    IMAP_LO = "lo"
    IMAP_ULTRA = "ultra"
    CODICE = "codice"
    SWE = "swe"
    HIT = "hit"
    SWAPI = "swapi"
    GLOWS = "glows"
    MAG = "mag"
