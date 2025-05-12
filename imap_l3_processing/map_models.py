from dataclasses import dataclass

import numpy as np

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
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.data.spectral_index_map_data.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.data.spectral_index_map_data.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.data.spectral_index_map_data.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.data.spectral_index_map_data.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.data.spectral_index_map_data.energy_delta_minus),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, self.data.spectral_index_map_data.energy_label),
            DataProductVariable(LATITUDE_VAR_NAME, self.data.spectral_index_map_data.latitude),
            DataProductVariable(LATITUDE_DELTA_VAR_NAME, self.data.coords.latitude_delta),
            DataProductVariable(LATITUDE_LABEL_VAR_NAME, self.data.coords.latitude_label),
            DataProductVariable(LONGITUDE_VAR_NAME, self.data.spectral_index_map_data.longitude),
            DataProductVariable(LONGITUDE_DELTA_VAR_NAME, self.data.coords.longitude_delta),
            DataProductVariable(LONGITUDE_LABEL_VAR_NAME, self.data.coords.longitude_label),
            DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, self.data.spectral_index_map_data.exposure_factor),
            DataProductVariable(OBS_DATE_VAR_NAME, self.data.spectral_index_map_data.obs_date),
            DataProductVariable(OBS_DATE_RANGE_VAR_NAME, self.data.spectral_index_map_data.obs_date_range),
            DataProductVariable(SOLID_ANGLE_VAR_NAME, self.data.spectral_index_map_data.solid_angle),
            DataProductVariable(ENA_SPECTRAL_INDEX_VAR_NAME, self.data.spectral_index_map_data.ena_spectral_index),
            DataProductVariable(ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME,
                                self.data.spectral_index_map_data.ena_spectral_index_stat_unc),
        ]
