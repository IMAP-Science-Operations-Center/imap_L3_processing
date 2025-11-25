from __future__ import annotations

import abc
import dataclasses
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
from imap_l3_processing.data_utils import safe_divide
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
ENA_INTENSITY_STAT_UNC_VAR_NAME = "ena_intensity_stat_uncert"
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
    ena_spectral_index_stat_unc: np.ndarray


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
        epoch=dataset[CoordNames.TIME.value].values,
        epoch_delta=dataset["epoch_delta"].values,
        energy=dataset.coords[CoordNames.ENERGY_L2.value].values,
        energy_delta_plus=dataset["energy_delta_plus"].values,
        energy_delta_minus=dataset["energy_delta_minus"].values,
        energy_label=dataset["energy_label"].values,
        latitude=dataset["latitude"].values,
        longitude=dataset["longitude"].values,
        exposure_factor=dataset["exposure_factor"].values,
        obs_date=dataset["obs_date"].values,
        obs_date_range=np.full_like(dataset["obs_date"].values, np.nan),
        solid_angle=np.full_like(dataset["latitude"].values, np.nan),
        ena_intensity=dataset["ena_intensity"].values,
        ena_intensity_stat_uncert=dataset["ena_intensity_stat_uncert"].values,
        ena_intensity_sys_err=np.full_like(dataset["ena_intensity_sys_err"].values, np.nan)
    )


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
                "ena_spectral_index_stat_unc": (full_shape, self.spectral_index_map_data.ena_spectral_index_stat_unc),
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
        DataProductVariable(ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, data.ena_spectral_index_stat_unc),
    ]


def _intensity_data_variables(data: IntensityMapData) -> list[DataProductVariable]:
    intensity_variables = [
        DataProductVariable(ENA_INTENSITY_VAR_NAME, data.ena_intensity),
        DataProductVariable(ENA_INTENSITY_STAT_UNC_VAR_NAME, data.ena_intensity_stat_uncert),
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


def combine_rectangular_intensity_map_data(maps: list[RectangularIntensityMapData],
                                           exposure_weighted: bool = True) -> RectangularIntensityMapData:
    for m in maps[1:]:
        assert np.array_equal(maps[0].coords.latitude_delta, m.coords.latitude_delta)
        assert np.array_equal(maps[0].coords.longitude_delta, m.coords.longitude_delta)
        assert np.array_equal(maps[0].coords.latitude_label, m.coords.latitude_label)
        assert np.array_equal(maps[0].coords.longitude_label, m.coords.longitude_label)
    intensity_map_data = combine_intensity_map_data([m.intensity_map_data for m in maps],
                                                    exposure_weighted=exposure_weighted)
    return RectangularIntensityMapData(intensity_map_data=intensity_map_data, coords=maps[0].coords)


def combine_healpix_intensity_map_data(maps: list[HealPixIntensityMapData],
                                       exposure_weighted: bool = True) -> HealPixIntensityMapData:
    for m in maps[1:]:
        assert np.array_equal(maps[0].coords.pixel_index, m.coords.pixel_index)
        assert np.array_equal(maps[0].coords.pixel_index_label, m.coords.pixel_index_label)
    intensity_map_data = combine_intensity_map_data([m.intensity_map_data for m in maps],
                                                    exposure_weighted=exposure_weighted)
    return HealPixIntensityMapData(intensity_map_data=intensity_map_data, coords=maps[0].coords)


def combine_intensity_map_data(maps: list[IntensityMapData], exposure_weighted: bool = True) -> IntensityMapData:
    first_map = maps[0]

    first_map_dict = dataclasses.asdict(first_map)

    fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_uncert", "ena_intensity_sys_err",
                               "bg_intensity", "bg_intensity_stat_uncert", "bg_intensity_sys_err",
                               "exposure_factor", "obs_date", "obs_date_range"}

    differing_fields = []
    for field in dataclasses.fields(first_map):
        if field.name not in fields_which_may_differ:
            differing_fields.append(field.name)
            supports_nan = np.issubdtype(first_map_dict[field.name].dtype, np.floating)
            assert np.all(
                [np.array_equal(dataclasses.asdict(m)[field.name], first_map_dict[field.name], equal_nan=supports_nan)
                 for m in maps]), f"{field.name}"

    intensities = np.array([m.ena_intensity for m in maps])
    intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
    intensity_stat_unc = np.array([m.ena_intensity_stat_uncert for m in maps])
    exposures = np.array([m.exposure_factor for m in maps])

    mask = np.isnan(intensities) | (exposures == 0) | np.isnan(exposures)

    intensities = np.where(mask, 0, intensities)
    intensity_sys_err = np.where(mask, 0, intensity_sys_err)
    intensity_stat_unc = np.where(mask, 0, intensity_stat_unc)

    masked_exposures = np.where(mask, 0, exposures)
    summed_exposures = np.sum(masked_exposures, axis=0)
    weights = exposures if exposure_weighted else np.full_like(exposures, 1)
    masked_weights = np.where(mask, 0, weights)

    combined_intensity_stat_uncert = calculated_weighted_uncertainty(intensity_stat_unc, masked_weights)
    combined_intensity_sys_err = calculated_weighted_uncertainty(intensity_sys_err, masked_weights)

    summed_intensity = np.sum(intensities * masked_weights, axis=0)
    exposure_weighted_summed_intensity = safe_divide(summed_intensity, np.sum(masked_weights, axis=0))

    avg_obs_date = calculate_datetime_weighted_average(np.ma.array([m.obs_date for m in maps]), masked_weights, 0)

    return dataclasses.replace(first_map,
                               ena_intensity=exposure_weighted_summed_intensity,
                               exposure_factor=summed_exposures,
                               ena_intensity_sys_err=combined_intensity_sys_err,
                               ena_intensity_stat_uncert=combined_intensity_stat_uncert,
                               obs_date=avg_obs_date
                               )


def calculated_weighted_uncertainty(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    masked_values = np.where(weights == 0, 0, values)
    numerator = np.sum(np.square(masked_values * weights), axis=0)
    return np.sqrt(safe_divide(numerator, np.square(np.sum(weights, axis=0))))


def calculate_datetime_weighted_average(data: np.ndarray, weights: np.ndarray, axis: int, **kwargs) -> np.ndarray:
    epoch_based_dates = np.array((np.ma.getdata(data) - TT2000_EPOCH) / timedelta(seconds=1),
                                 dtype=float)

    averaged_dates_as_seconds = np.ma.average(epoch_based_dates, weights=weights,
                                              axis=axis, **kwargs)

    return np.ma.array(
        averaged_dates_as_seconds.data * timedelta(seconds=1) + TT2000_EPOCH,
        mask=averaged_dates_as_seconds.mask,
    )


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
    epoch_delta: np.ndarray
    epoch_j2000: np.ndarray
    repointing: int
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray
