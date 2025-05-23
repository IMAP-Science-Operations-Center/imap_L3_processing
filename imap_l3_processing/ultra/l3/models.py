from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS


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
                epoch=cdf[CoordNames.TIME.value][0],
                energy=read_numeric_variable(cdf[CoordNames.ENERGY_ULTRA.value]),
                exposure=read_numeric_variable(cdf["exposure_time"]),
                latitude=read_numeric_variable(cdf[CoordNames.ELEVATION_L1C.value]),
                longitude=read_numeric_variable(cdf[CoordNames.AZIMUTH_L1C.value]),
                healpix_index=cdf[CoordNames.HEALPIX_INDEX.value][...],
                sensitivity=read_numeric_variable(cdf["sensitivity"]),
            )

    def to_xarray(self):
        return xr.Dataset(
            {
                "counts": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    self.counts,
                ),
                "exposure_time": (
                    [CoordNames.TIME.value,
                     CoordNames.ENERGY_ULTRA.value,
                     CoordNames.HEALPIX_INDEX.value],
                    self.exposure,
                ),
                "sensitivity": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY_ULTRA.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    self.sensitivity,
                ),
                CoordNames.AZIMUTH_L1C.value: (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.longitude,
                ),
                CoordNames.ELEVATION_L1C.value: (
                    [CoordNames.HEALPIX_INDEX.value],
                    self.latitude,
                ),
            },
            coords={
                CoordNames.TIME.value: [
                    (self.epoch - TT2000_EPOCH).total_seconds() * ONE_SECOND_IN_NANOSECONDS,
                ],
                CoordNames.ENERGY_ULTRA.value: self.energy,
                CoordNames.HEALPIX_INDEX.value: self.healpix_index,
            }
        )
