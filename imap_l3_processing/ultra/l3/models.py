from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable


@dataclass
class UltraGlowsL3eData:
    epoch: datetime
    energy: ndarray
    latitude: ndarray
    longitude: ndarray
    healpix_index: ndarray
    survival_probability: ndarray

    @classmethod
    def read_from_path(cls, path_to_cdf: Path) -> Self:
        with CDF(str(path_to_cdf)) as cdf:
            return UltraGlowsL3eData(
                epoch=cdf["epoch"][0],
                energy=read_numeric_variable(cdf["energy"]),
                latitude=read_numeric_variable(cdf["latitude"]),
                longitude=read_numeric_variable(cdf["longitude"]),
                healpix_index=cdf["healpix_index"][...],
                survival_probability=read_numeric_variable(cdf["probability_of_survival"]),
            )


@dataclass
class UltraL1CPSet:
    epoch: datetime
    energy: ndarray
    counts: ndarray
    exposure: ndarray
    healpix_index: ndarray
    latitude: ndarray
    longitude: ndarray
    sensitivity: ndarray

    @classmethod
    def read_from_path(cls, path: Path) -> Self:
        with CDF(str(path)) as cdf:
            return UltraL1CPSet(
                counts=read_numeric_variable(cdf["counts"]),
                epoch=cdf["epoch"][0],
                energy=read_numeric_variable(cdf["energy"]),
                exposure=read_numeric_variable(cdf["exposure_time"]),
                latitude=read_numeric_variable(cdf["latitude"]),
                longitude=read_numeric_variable(cdf["longitude"]),
                healpix_index=cdf["healpix_index"][...],
                sensitivity=read_numeric_variable(cdf["sensitivity"]),
            )
