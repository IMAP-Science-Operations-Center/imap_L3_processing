from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from numpy import ndarray
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable, read_variable_and_mask_fill_values
from imap_l3_processing.constants import TT2000_EPOCH, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.models import DataProduct, DataProductVariable, IntensityMapData, HealPixCoords

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
ENERGY_LABEL_VAR_NAME = "energy_label"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
PIXEL_INDEX_VAR_NAME = "pixel_index"
PIXEL_INDEX_LABEL_VAR_NAME = "pixel_index_label"
LATITUDE_VAR_NAME = "latitude"
LATITUDE_LABEL_VAR_NAME = "latitude_label"
LATITUDE_DELTA_VAR_NAME = "latitude_delta"
LONGITUDE_VAR_NAME = "longitude"
LONGITUDE_LABEL_VAR_NAME = "longitude_label"
LONGITUDE_DELTA_VAR_NAME = "longitude_delta"
ENA_INTENSITY_VAR_NAME = "ena_intensity"
ENA_INTENSITY_STAT_UNC_VAR_NAME = "ena_intensity_stat_unc"
ENA_INTENSITY_SYS_ERR_VAR_NAME = "ena_intensity_sys_err"
EXPOSURE_FACTOR_VAR_NAME = "exposure_factor"
OBS_DATE_VAR_NAME = "obs_date"
OBS_DATE_RANGE_VAR_NAME = "obs_date_range"
SOLID_ANGLE_VAR_NAME = "solid_angle"


@dataclass
class UltraL2Map(IntensityMapData, HealPixCoords):
    @classmethod
    def read_from_path(cls, file_path: Path):
        with CDF(str(file_path)) as cdf:
            return UltraL2Map(
                epoch=read_variable_and_mask_fill_values(cdf["epoch"]),
                epoch_delta=read_variable_and_mask_fill_values(cdf["epoch_delta"]),
                energy=read_numeric_variable(cdf["energy"]),
                energy_delta_plus=read_numeric_variable(cdf["energy_delta_plus"]),
                energy_delta_minus=read_numeric_variable(cdf["energy_delta_minus"]),
                energy_label=cdf["energy_label"][...],
                latitude=read_numeric_variable(cdf["latitude"]),
                longitude=read_numeric_variable(cdf["longitude"]),
                exposure_factor=read_numeric_variable(cdf["exposure_factor"]),
                obs_date=read_variable_and_mask_fill_values(cdf["obs_date"]),
                obs_date_range=read_variable_and_mask_fill_values(cdf["obs_date_range"]),
                solid_angle=read_numeric_variable(cdf["solid_angle"]),
                ena_intensity=read_numeric_variable(cdf["ena_intensity"]),
                ena_intensity_stat_unc=read_numeric_variable(cdf["ena_intensity_stat_unc"]),
                ena_intensity_sys_err=read_numeric_variable(cdf["ena_intensity_sys_err"]),
                pixel_index=read_numeric_variable(cdf["pixel_index"]),
                pixel_index_label=cdf["pixel_index_label"][...]
            )

    @property
    def nside(self) -> int:
        return int(np.sqrt(len(self.pixel_index) / 12))


@dataclass
class UltraL3SurvivalCorrectedDataProduct(DataProduct, IntensityMapData, HealPixCoords):
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, self.energy_label),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(PIXEL_INDEX_VAR_NAME, self.pixel_index),
            DataProductVariable(PIXEL_INDEX_LABEL_VAR_NAME, self.pixel_index_label),
            DataProductVariable(LATITUDE_VAR_NAME, self.latitude),
            DataProductVariable(LONGITUDE_VAR_NAME, self.longitude),
            DataProductVariable(ENA_INTENSITY_VAR_NAME, self.ena_intensity),
            DataProductVariable(ENA_INTENSITY_STAT_UNC_VAR_NAME, self.ena_intensity_stat_unc),
            DataProductVariable(ENA_INTENSITY_SYS_ERR_VAR_NAME, self.ena_intensity_sys_err),
            DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, self.exposure_factor),
            DataProductVariable(OBS_DATE_VAR_NAME, self.obs_date),
            DataProductVariable(OBS_DATE_RANGE_VAR_NAME, self.obs_date_range),
            DataProductVariable(SOLID_ANGLE_VAR_NAME, self.solid_angle),
        ]


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

    def to_xarray(self):
        return xr.Dataset(
            {
                "counts": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY.value,
                        CoordNames.HEALPIX_INDEX.value,
                    ],
                    self.counts,
                ),
                "exposure_time": (
                    [CoordNames.TIME.value,
                     CoordNames.ENERGY.value,
                     CoordNames.HEALPIX_INDEX.value],
                    self.exposure,
                ),
                "sensitivity": (
                    [
                        CoordNames.TIME.value,
                        CoordNames.ENERGY.value,
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
                CoordNames.ENERGY.value: self.energy,
                CoordNames.HEALPIX_INDEX.value: self.healpix_index,
            }
        )
