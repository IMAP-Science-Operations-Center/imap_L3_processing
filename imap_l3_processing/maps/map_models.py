from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import timedelta, datetime
from pathlib import Path
from typing import Generic, Optional

import numpy as np
import xarray
from imap_processing.ena_maps.ena_maps import HealpixSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable_and_mask_fill_values, read_numeric_variable
from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.models import DataProduct, DataProductVariable, D

EPOCH_VAR_NAME = "epoch"
EPOCH_DELTA_VAR_NAME = "epoch_delta"
ENERGY_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_VAR_NAME = "energy_delta_minus"
ENERGY_LABEL_VAR_NAME = "energy_label"

LONGITUDE_VAR_NAME = "longitude"
LONGITUDE_DELTA_VAR_NAME = "longitude_delta"
LONGITUDE_LABEL_VAR_NAME = "longitude_label"
LATITUDE_VAR_NAME = "latitude"
LATITUDE_DELTA_VAR_NAME = "latitude_delta"
LATITUDE_LABEL_VAR_NAME = "latitude_label"

EXPOSURE_FACTOR_VAR_NAME = "exposure_factor"
OBS_DATE_VAR_NAME = "obs_date"
OBS_DATE_RANGE_VAR_NAME = "obs_date_range"
SOLID_ANGLE_VAR_NAME = "solid_angle"
ENA_SPECTRAL_INDEX_VAR_NAME = "ena_spectral_index"
ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME = "ena_spectral_index_stat_uncert"

ENA_INTENSITY_VAR_NAME = "ena_intensity"
ENA_INTENSITY_STAT_UNCERT_VAR_NAME = "ena_intensity_stat_uncert"
ENA_INTENSITY_SYS_ERR_VAR_NAME = "ena_intensity_sys_err"

BG_INTENSITY_VAR_NAME = "bg_intensity"
BG_INTENSITY_STAT_UNC_VAR_NAME = "bg_intensity_stat_uncert"
BG_INTENSITY_SYS_ERR_VAR_NAME = "bg_intensity_sys_err"

PIXEL_INDEX_VAR_NAME = "pixel_index"
PIXEL_INDEX_LABEL_VAR_NAME = "pixel_index_label"


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

    @property
    def nside(self) -> int:
        return int(np.sqrt(len(self.pixel_index) / 12))


@dataclass
class RectangularCoords:
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray


@dataclass
class IntensityMapData(MapData):
    ena_intensity: np.ndarray
    ena_intensity_stat_uncert: np.ndarray
    ena_intensity_sys_err: np.ndarray
    bg_intensity: Optional[np.ndarray] = None
    bg_intensity_stat_uncert: Optional[np.ndarray] = None
    bg_intensity_sys_err: Optional[np.ndarray] = None


@dataclass
class SpectralIndexMapData(MapData):
    ena_spectral_index: np.ndarray
    ena_spectral_index_stat_uncert: np.ndarray


@dataclass
class RectangularIntensityMapData:
    intensity_map_data: IntensityMapData
    coords: RectangularCoords

    @classmethod
    def read_from_path(cls, cdf_path: Path | str) -> RectangularIntensityMapData:
        with CDF(str(cdf_path)) as cdf:
            return RectangularIntensityMapData(
                intensity_map_data=_read_intensity_map_data_from_open_cdf(cdf),
                coords=_read_rectangular_coords_from_open_cdf(cdf),
            )


@dataclass
class RectangularSpectralIndexMapData:
    spectral_index_map_data: SpectralIndexMapData
    coords: RectangularCoords


@dataclass
class HealPixIntensityMapData:
    intensity_map_data: IntensityMapData
    coords: HealPixCoords

    @classmethod
    def read_from_path(cls, cdf_path: Path | str) -> HealPixIntensityMapData:
        with CDF(str(cdf_path)) as cdf:
            return HealPixIntensityMapData(
                intensity_map_data=_read_intensity_map_data_from_open_cdf(cdf),
                coords=_read_healpix_coords_from_open_cdf(cdf),
            )

    def to_healpix_skymap(self) -> HealpixSkyMap:
        healpix_map = HealpixSkyMap(self.coords.nside, SpiceFrame.ECLIPJ2000)

        full_shape = [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.HEALPIX_INDEX.value]
        healpix_map.data_1d = xarray.Dataset(
            data_vars={
                "latitude": ([CoordNames.HEALPIX_INDEX.value], self.intensity_map_data.latitude),
                "longitude": ([CoordNames.HEALPIX_INDEX.value], self.intensity_map_data.longitude),
                "solid_angle": ([CoordNames.HEALPIX_INDEX.value], self.intensity_map_data.solid_angle),
                "obs_date_range": (full_shape, self.intensity_map_data.obs_date_range),
                "obs_date": (full_shape, self.intensity_map_data.obs_date),
                "exposure_factor": (full_shape, self.intensity_map_data.exposure_factor),
                "ena_intensity": (full_shape, self.intensity_map_data.ena_intensity),
                "ena_intensity_stat_uncert": (full_shape, self.intensity_map_data.ena_intensity_stat_uncert),
                "ena_intensity_sys_err": (full_shape, self.intensity_map_data.ena_intensity_sys_err),
            },
            coords={
                CoordNames.TIME.value: self.intensity_map_data.epoch,
                CoordNames.ENERGY_L2.value: self.intensity_map_data.energy,
                CoordNames.HEALPIX_INDEX.value: self.coords.pixel_index,
            })

        healpix_map.data_1d = healpix_map.data_1d \
            .assign({"obs_date": (full_shape, healpix_map.data_1d["obs_date"].values.astype(np.float64))}) \
            .rename({CoordNames.HEALPIX_INDEX.value: CoordNames.GENERIC_PIXEL.value})

        return healpix_map

    @classmethod
    def read_from_xarray(cls, input_dataset):
        return HealPixIntensityMapData(
            intensity_map_data=_read_intensity_map_data_from_xarray(input_dataset),
            coords=_read_healpix_coords_from_xarray(input_dataset),
        )


@dataclass
class SpectralIndexDependencies(metaclass=abc.ABCMeta):
    map_data: RectangularIntensityMapData | HealPixIntensityMapData

    @abc.abstractmethod
    def get_fit_energy_ranges(self) -> np.ndarray:
        raise NotImplementedError


def convert_tt2000_time_to_datetime(time: np.ndarray) -> np.ndarray:
    return time / 1e9 * timedelta(seconds=1) + TT2000_EPOCH


def _read_intensity_map_data_from_open_cdf(cdf: CDF) -> IntensityMapData:
    masked_obs_date = read_variable_and_mask_fill_values(cdf["obs_date"])
    if np.issubdtype(cdf["obs_date"].dtype, np.number):
        obs_date = convert_tt2000_time_to_datetime(masked_obs_date.filled(0))
        masked_obs_date = np.ma.masked_array(data=obs_date, mask=masked_obs_date.mask)

    map_intensity_data = IntensityMapData(epoch=cdf["epoch"][...],
                                          epoch_delta=read_variable_and_mask_fill_values(cdf["epoch_delta"]),
                                          energy=read_numeric_variable(cdf["energy"]),
                                          energy_delta_plus=read_numeric_variable(cdf["energy_delta_plus"]),
                                          energy_delta_minus=read_numeric_variable(cdf["energy_delta_minus"]),
                                          energy_label=cdf["energy_label"][...],
                                          latitude=read_numeric_variable(cdf["latitude"]),
                                          longitude=read_numeric_variable(cdf["longitude"]),
                                          exposure_factor=read_numeric_variable(cdf["exposure_factor"]),
                                          obs_date=masked_obs_date,
                                          obs_date_range=read_variable_and_mask_fill_values(cdf["obs_date_range"]),
                                          solid_angle=read_numeric_variable(cdf["solid_angle"]),
                                          ena_intensity=read_numeric_variable(cdf["ena_intensity"]),
                                          ena_intensity_stat_uncert=read_numeric_variable(
                                              cdf["ena_intensity_stat_uncert"]),
                                          ena_intensity_sys_err=read_numeric_variable(cdf["ena_intensity_sys_err"]), )

    if "bg_intensity" in cdf:
        map_intensity_data.bg_intensity = read_numeric_variable(cdf["bg_intensity"])
        map_intensity_data.bg_intensity_sys_err = read_numeric_variable(cdf["bg_intensity_sys_err"])
        map_intensity_data.bg_intensity_stat_uncert = read_numeric_variable(cdf["bg_intensity_stat_uncert"])

    return map_intensity_data


def _read_healpix_coords_from_open_cdf(cdf: CDF) -> HealPixCoords:
    return HealPixCoords(
        pixel_index=cdf["pixel_index"][...],
        pixel_index_label=cdf["pixel_index_label"][...]
    )


def _read_intensity_map_data_from_xarray(dataset: xarray.Dataset) -> IntensityMapData:
    return IntensityMapData(
        epoch=_replace_fill_values_in_xarray(dataset, CoordNames.TIME.value),
        epoch_delta=_replace_fill_values_in_xarray(dataset, "epoch_delta"),
        energy=dataset.coords[CoordNames.ENERGY_L2.value].values,
        energy_delta_plus=_replace_fill_values_in_xarray(dataset, "energy_delta_plus"),
        energy_delta_minus=_replace_fill_values_in_xarray(dataset, "energy_delta_minus"),
        energy_label=_replace_fill_values_in_xarray(dataset, "energy_label"),
        latitude=_replace_fill_values_in_xarray(dataset, "latitude"),
        longitude=_replace_fill_values_in_xarray(dataset, "longitude"),
        exposure_factor=_replace_fill_values_in_xarray(dataset, "exposure_factor"),
        obs_date=_replace_fill_values_in_xarray(dataset, "obs_date"),
        obs_date_range=_replace_fill_values_in_xarray(dataset, "obs_date_range"),
        solid_angle=_replace_fill_values_in_xarray(dataset, "solid_angle"),
        ena_intensity=_replace_fill_values_in_xarray(dataset, "ena_intensity"),
        ena_intensity_stat_uncert=_replace_fill_values_in_xarray(dataset, "ena_intensity_stat_uncert"),
        ena_intensity_sys_err=_replace_fill_values_in_xarray(dataset, "ena_intensity_sys_err")
    )


def _replace_fill_values_in_xarray(dataset, variable):
    if 'FILLVAL' in dataset[variable].attrs:
        return np.where(dataset[variable].values == dataset[variable].attrs['FILLVAL'], np.nan,
                        dataset[variable].values)
    else:
        return dataset[variable].values

def _read_healpix_coords_from_xarray(dataset: xarray.Dataset) -> HealPixCoords:
    return HealPixCoords(
        pixel_index=dataset[CoordNames.HEALPIX_INDEX.value].values,
        pixel_index_label=dataset["pixel_index_label"].values
    )


def _read_rectangular_coords_from_open_cdf(cdf: CDF) -> RectangularCoords:
    return RectangularCoords(
        latitude_delta=cdf["latitude_delta"][...],
        latitude_label=cdf["latitude_label"][...],
        longitude_delta=cdf["longitude_delta"][...],
        longitude_label=cdf["longitude_label"][...],
    )


@dataclass
class HealPixSpectralIndexMapData:
    spectral_index_map_data: SpectralIndexMapData
    coords: HealPixCoords

    def to_healpix_skymap(self) -> HealpixSkyMap:
        healpix_map = HealpixSkyMap(self.coords.nside, SpiceFrame.ECLIPJ2000)

        full_shape = [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, CoordNames.HEALPIX_INDEX.value]
        healpix_map.data_1d = xarray.Dataset(
            data_vars={
                "latitude": ([CoordNames.HEALPIX_INDEX.value], self.spectral_index_map_data.latitude),
                "longitude": ([CoordNames.HEALPIX_INDEX.value], self.spectral_index_map_data.longitude),
                "solid_angle": ([CoordNames.HEALPIX_INDEX.value], self.spectral_index_map_data.solid_angle),
                "obs_date_range": (full_shape, self.spectral_index_map_data.obs_date_range),
                "obs_date": (full_shape, self.spectral_index_map_data.obs_date),
                "exposure_factor": (full_shape, self.spectral_index_map_data.exposure_factor),

                "ena_spectral_index": (full_shape, self.spectral_index_map_data.ena_spectral_index),
                "ena_spectral_index_stat_uncert": (
                    full_shape, self.spectral_index_map_data.ena_spectral_index_stat_uncert),
            },
            coords={
                CoordNames.TIME.value: self.spectral_index_map_data.epoch,
                CoordNames.ENERGY_L2.value: self.spectral_index_map_data.energy,
                CoordNames.HEALPIX_INDEX.value: self.coords.pixel_index,
            })

        healpix_map.data_1d = healpix_map.data_1d \
            .assign({"obs_date": (full_shape, healpix_map.data_1d["obs_date"].values.astype(np.float64))}) \
            .rename({CoordNames.HEALPIX_INDEX.value: CoordNames.GENERIC_PIXEL.value})

        return healpix_map


@dataclass
class MapDataProduct(DataProduct[D], Generic[D]):
    data: D

    @abc.abstractmethod
    def to_data_product_variables(self) -> list[DataProductVariable]:
        raise NotImplementedError


class HealPixSpectralIndexDataProduct(MapDataProduct[HealPixSpectralIndexMapData]):
    data: HealPixSpectralIndexMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _spectral_index_data_variables(self.data.spectral_index_map_data) \
            + _healpix_coords_to_variables(self.data.coords)


class HealPixIntensityDataProduct(MapDataProduct[HealPixIntensityMapData]):
    data: HealPixIntensityMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _intensity_data_variables(self.data.intensity_map_data) \
            + _healpix_coords_to_variables(self.data.coords)


class RectangularSpectralIndexDataProduct(MapDataProduct[RectangularSpectralIndexMapData]):
    data: RectangularSpectralIndexMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _spectral_index_data_variables(self.data.spectral_index_map_data) \
            + _rectangular_coords_to_variables(self.data.coords)


class RectangularIntensityDataProduct(MapDataProduct[RectangularIntensityMapData]):
    data: RectangularIntensityMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return _intensity_data_variables(self.data.intensity_map_data) \
            + _rectangular_coords_to_variables(self.data.coords)


def _map_data_to_variables(data: MapData) -> list[DataProductVariable]:
    return [
        DataProductVariable(EPOCH_VAR_NAME, data.epoch),
        DataProductVariable(EPOCH_DELTA_VAR_NAME, data.epoch_delta),
        DataProductVariable(ENERGY_VAR_NAME, data.energy),
        DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, data.energy_delta_plus),
        DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, data.energy_delta_minus),
        DataProductVariable(ENERGY_LABEL_VAR_NAME, data.energy_label),
        DataProductVariable(LATITUDE_VAR_NAME, data.latitude),
        DataProductVariable(LONGITUDE_VAR_NAME, data.longitude),
        DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, data.exposure_factor),
        DataProductVariable(OBS_DATE_VAR_NAME, data.obs_date),
        DataProductVariable(OBS_DATE_RANGE_VAR_NAME, data.obs_date_range),
        DataProductVariable(SOLID_ANGLE_VAR_NAME, data.solid_angle),
    ]


def _spectral_index_data_variables(data: SpectralIndexMapData) -> list[DataProductVariable]:
    return _map_data_to_variables(data) + [
        DataProductVariable(ENA_SPECTRAL_INDEX_VAR_NAME, data.ena_spectral_index),
        DataProductVariable(ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, data.ena_spectral_index_stat_uncert),
    ]


def _intensity_data_variables(data: IntensityMapData) -> list[DataProductVariable]:
    intensity_variables = [
        DataProductVariable(ENA_INTENSITY_VAR_NAME, data.ena_intensity),
        DataProductVariable(ENA_INTENSITY_STAT_UNCERT_VAR_NAME, data.ena_intensity_stat_uncert),
        DataProductVariable(ENA_INTENSITY_SYS_ERR_VAR_NAME, data.ena_intensity_sys_err),
    ]
    if data.bg_intensity is not None:
        intensity_variables.extend([
            DataProductVariable(BG_INTENSITY_VAR_NAME, data.bg_intensity),
            DataProductVariable(BG_INTENSITY_STAT_UNC_VAR_NAME, data.bg_intensity_stat_uncert),
            DataProductVariable(BG_INTENSITY_SYS_ERR_VAR_NAME, data.bg_intensity_sys_err),
        ])

    return _map_data_to_variables(data) + intensity_variables


def _rectangular_coords_to_variables(coords: RectangularCoords) -> list[DataProductVariable]:
    return [
        DataProductVariable(LATITUDE_DELTA_VAR_NAME, coords.latitude_delta),
        DataProductVariable(LATITUDE_LABEL_VAR_NAME, coords.latitude_label),
        DataProductVariable(LONGITUDE_DELTA_VAR_NAME, coords.longitude_delta),
        DataProductVariable(LONGITUDE_LABEL_VAR_NAME, coords.longitude_label)
    ]


def _healpix_coords_to_variables(coords: HealPixCoords) -> list[DataProductVariable]:
    return [
        DataProductVariable(PIXEL_INDEX_VAR_NAME, coords.pixel_index),
        DataProductVariable(PIXEL_INDEX_LABEL_VAR_NAME, coords.pixel_index_label),
    ]


def calculate_datetime_weighted_average(data: np.ndarray, weights: np.ndarray, axis: int, **kwargs) -> np.ndarray:
    if isinstance(np.ravel(np.ma.getdata(data))[0], datetime):

        epoch_based_dates = np.array((np.ma.getdata(data) - TT2000_EPOCH) / timedelta(seconds=1),
                                     dtype=float)

        averaged_dates_as_seconds = np.ma.average(epoch_based_dates, weights=weights,
                                                  axis=axis, **kwargs)

        return np.ma.array(
            averaged_dates_as_seconds.data * timedelta(seconds=1) + TT2000_EPOCH,
            mask=averaged_dates_as_seconds.mask,
        )
    else:
        return np.ma.average(data, weights=weights, axis=axis, **kwargs)


@dataclass
class GlowsL3eRectangularMapInputData:
    epoch: datetime
    epoch_j2000: np.ndarray
    repointing: int
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray


@dataclass
class InputRectangularPointingSet:
    epoch: datetime
    epoch_delta: Optional[np.ndarray]
    epoch_j2000: np.ndarray
    repointing: int
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray
    pointing_start_met: Optional[float]
    pointing_end_met: Optional[float]
