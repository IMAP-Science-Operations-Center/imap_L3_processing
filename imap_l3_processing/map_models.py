import dataclasses
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from imap_l3_processing.constants import TT2000_EPOCH
from imap_l3_processing.data_utils import safe_divide
from imap_l3_processing.models import DataProduct, DataProductVariable

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
ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME = "ena_spectral_index_stat_unc"

ENA_INTENSITY_VAR_NAME = "ena_intensity"
ENA_INTENSITY_STAT_UNC_VAR_NAME = "ena_intensity_stat_unc"
ENA_INTENSITY_SYS_ERR_VAR_NAME = "ena_intensity_sys_err"


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


@dataclass
class RectangularCoords:
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray


@dataclass
class IntensityMapData(MapData):
    ena_intensity: np.ndarray
    ena_intensity_stat_unc: np.ndarray
    ena_intensity_sys_err: np.ndarray


@dataclass
class SpectralIndexMapData(MapData):
    ena_spectral_index: np.ndarray
    ena_spectral_index_stat_unc: np.ndarray


@dataclass
class RectangularIntensityMapData:
    intensity_map_data: IntensityMapData
    coords: RectangularCoords


@dataclass
class RectangularSpectralIndexMapData:
    spectral_index_map_data: SpectralIndexMapData
    coords: RectangularCoords


@dataclass
class RectangularSpectralIndexDataProduct(DataProduct):
    data: RectangularSpectralIndexMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        spectral_index_variables = [
            DataProductVariable(ENA_SPECTRAL_INDEX_VAR_NAME, self.data.spectral_index_map_data.ena_spectral_index),
            DataProductVariable(ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME,
                                self.data.spectral_index_map_data.ena_spectral_index_stat_unc),
        ]
        return _map_data_to_variables(self.data.spectral_index_map_data) \
            + spectral_index_variables \
            + _rectangular_coords_to_variables(self.data.coords)


@dataclass
class RectangularIntensityDataProduct(DataProduct):
    data: RectangularIntensityMapData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        intensity_variables = [
            DataProductVariable(ENA_INTENSITY_VAR_NAME, self.data.intensity_map_data.ena_intensity),
            DataProductVariable(ENA_INTENSITY_STAT_UNC_VAR_NAME,
                                self.data.intensity_map_data.ena_intensity_stat_unc),
            DataProductVariable(ENA_INTENSITY_SYS_ERR_VAR_NAME, self.data.intensity_map_data.ena_intensity_sys_err),

        ]
        return _map_data_to_variables(self.data.intensity_map_data) \
            + intensity_variables \
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


def _rectangular_coords_to_variables(coords: RectangularCoords) -> list[DataProductVariable]:
    return [
        DataProductVariable(LATITUDE_DELTA_VAR_NAME, coords.latitude_delta),
        DataProductVariable(LATITUDE_LABEL_VAR_NAME, coords.latitude_label),
        DataProductVariable(LONGITUDE_DELTA_VAR_NAME, coords.longitude_delta),
        DataProductVariable(LONGITUDE_LABEL_VAR_NAME, coords.longitude_label)
    ]


def combine_rectangular_intensity_map_data(maps: list[RectangularIntensityMapData]) -> RectangularIntensityMapData:
    for m in maps[1:]:
        assert np.array_equal(maps[0].coords.latitude_delta, m.coords.latitude_delta)
        assert np.array_equal(maps[0].coords.longitude_delta, m.coords.longitude_delta)
        assert np.array_equal(maps[0].coords.latitude_label, m.coords.latitude_label)
        assert np.array_equal(maps[0].coords.longitude_label, m.coords.longitude_label)
    intensity_map_data = combine_intensity_map_data([m.intensity_map_data for m in maps])
    return RectangularIntensityMapData(
        intensity_map_data=intensity_map_data,
        coords=maps[0].coords)


def combine_intensity_map_data(maps: list[IntensityMapData]) -> IntensityMapData:
    first_map = maps[0]

    first_map_dict = dataclasses.asdict(first_map)

    fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_unc", "ena_intensity_sys_err",
                               "exposure_factor", "obs_date", "obs_date_range"}

    differing_fields = []
    for field in dataclasses.fields(first_map):
        if field.name not in fields_which_may_differ:
            differing_fields.append(field.name)
            assert np.all(
                [dataclasses.asdict(m)[field.name] == first_map_dict[field.name] for m in maps]), f"{field.name}"

    intensities = np.array([m.ena_intensity for m in maps])
    intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
    intensity_stat_unc = np.array([m.ena_intensity_stat_unc for m in maps])
    exposures = np.array([m.exposure_factor for m in maps])

    intensities = np.where(exposures == 0, 0, intensities)
    intensity_sys_err = np.where(exposures == 0, 0, intensity_sys_err)
    intensity_stat_unc = np.where(exposures == 0, 0, intensity_stat_unc)

    combined_intensity_stat_unc = np.sqrt(
        safe_divide(np.sum(np.square(intensity_stat_unc * exposures), axis=0),
                    np.square(np.sum(exposures, axis=0)))
    )
    summed_exposures = np.sum(exposures, axis=0)
    ena_intensity = np.ma.average(intensities, weights=exposures, axis=0).filled(np.nan)
    ena_intensity_sys_err = np.ma.average(intensity_sys_err, weights=exposures, axis=0).filled(np.nan)

    avg_obs_date = calculate_datetime_weighted_average(np.ma.array([m.obs_date for m in maps]), exposures, 0)

    return dataclasses.replace(first_map,
                               ena_intensity=ena_intensity,
                               exposure_factor=summed_exposures,
                               ena_intensity_sys_err=ena_intensity_sys_err,
                               ena_intensity_stat_unc=combined_intensity_stat_unc,
                               obs_date=avg_obs_date
                               )


def calculate_datetime_weighted_average(data: np.ndarray, weights: np.ndarray, axis: int, **kwargs) -> np.ndarray:
    epoch_based_dates = np.array((np.ma.getdata(data) - TT2000_EPOCH) / timedelta(seconds=1),
                                 dtype=float)

    averaged_dates_as_seconds = np.ma.average(epoch_based_dates, weights=weights,
                                              axis=axis, **kwargs)

    return np.ma.array(
        averaged_dates_as_seconds.data * timedelta(seconds=1) + TT2000_EPOCH,
        mask=averaged_dates_as_seconds.mask,
    )
