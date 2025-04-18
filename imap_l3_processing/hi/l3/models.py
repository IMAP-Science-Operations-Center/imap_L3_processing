import dataclasses
from dataclasses import dataclass
from datetime import datetime

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
class HiMapData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    energy_label: np.ndarray
    latitude: np.ndarray
    latitude_delta: np.ndarray
    latitude_label: np.ndarray
    longitude: np.ndarray
    longitude_delta: np.ndarray
    longitude_label: np.ndarray
    exposure_factor: np.ndarray
    obs_date: np.ndarray
    obs_date_range: np.ndarray
    solid_angle: np.ndarray


@dataclass
class HiIntensityMapData(HiMapData):
    ena_intensity: np.ndarray
    ena_intensity_stat_unc: np.ndarray
    ena_intensity_sys_err: np.ndarray


class HiDataProduct(DataProduct, HiMapData):
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_VAR_NAME, self.epoch),
            DataProductVariable(EPOCH_DELTA_VAR_NAME, self.epoch_delta),
            DataProductVariable(ENERGY_VAR_NAME, self.energy),
            DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, self.energy_delta_plus),
            DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, self.energy_delta_minus),
            DataProductVariable(ENERGY_LABEL_VAR_NAME, self.energy_label),
            DataProductVariable(LATITUDE_VAR_NAME, self.latitude),
            DataProductVariable(LATITUDE_DELTA_VAR_NAME, self.latitude_delta),
            DataProductVariable(LATITUDE_LABEL_VAR_NAME, self.latitude_label),
            DataProductVariable(LONGITUDE_VAR_NAME, self.longitude),
            DataProductVariable(LONGITUDE_DELTA_VAR_NAME, self.longitude_delta),
            DataProductVariable(LONGITUDE_LABEL_VAR_NAME, self.longitude_label),
            DataProductVariable(EXPOSURE_FACTOR_VAR_NAME, self.exposure_factor),
            DataProductVariable(OBS_DATE_VAR_NAME, self.obs_date),
            DataProductVariable(OBS_DATE_RANGE_VAR_NAME, self.obs_date_range),
            DataProductVariable(SOLID_ANGLE_VAR_NAME, self.solid_angle),
        ]


@dataclass
class HiL3SpectralIndexDataProduct(HiDataProduct):
    ena_spectral_index: np.ndarray
    ena_spectral_index_stat_unc: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return super().to_data_product_variables() + [
            DataProductVariable(ENA_SPECTRAL_INDEX_VAR_NAME, self.ena_spectral_index),
            DataProductVariable(ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, self.ena_spectral_index_stat_unc),
        ]


@dataclass
class HiL3SurvivalCorrectedDataProduct(HiDataProduct, HiIntensityMapData):
    def to_data_product_variables(self) -> list[DataProductVariable]:
        return super().to_data_product_variables() + [
            DataProductVariable(ENA_INTENSITY_VAR_NAME, self.ena_intensity),
            DataProductVariable(ENA_INTENSITY_STAT_UNC_VAR_NAME, self.ena_intensity_stat_unc),
            DataProductVariable(ENA_INTENSITY_SYS_ERR_VAR_NAME, self.ena_intensity_sys_err),
        ]


@dataclass
class HiL1cData:
    epoch: datetime
    epoch_j2000: np.ndarray
    exposure_times: np.ndarray
    esa_energy_step: np.ndarray


@dataclass
class GlowsL3eData:
    epoch: datetime
    energy: np.ndarray
    spin_angle: np.ndarray
    probability_of_survival: np.ndarray


def combine_maps(maps: list[HiL3SurvivalCorrectedDataProduct]) -> HiL3SurvivalCorrectedDataProduct:
    first_map = maps[0]

    first_map_dict = dataclasses.asdict(first_map)

    fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_unc", "ena_intensity_sys_err",
                               "exposure_factor", "obs_date", "obs_date_range", "parent_file_names", "input_metadata"}
    for field in dataclasses.fields(first_map):
        if field.name not in fields_which_may_differ:
            assert np.all(
                [dataclasses.asdict(m)[field.name] == first_map_dict[field.name] for m in maps]), f"{field.name}"

    intensities = np.array([m.ena_intensity for m in maps])
    intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
    intensity_stat_unc = np.array([m.ena_intensity_stat_unc for m in maps])
    exposures = np.array([m.exposure_factor for m in maps])

    intensities = np.where(exposures == 0, 0, intensities)
    intensity_sys_err = np.where(exposures == 0, 0, intensity_sys_err)
    intensity_stat_unc = np.where(exposures == 0, 0, intensity_stat_unc)

    combined_intensity_stat_unc = np.sqrt((np.sum(np.square(intensity_stat_unc * exposures), axis=0) /
                                           np.square(np.sum(exposures, axis=0))))

    return dataclasses.replace(first_map,
                               ena_intensity=np.average(intensities, axis=0, weights=exposures),
                               exposure_factor=np.sum(exposures, axis=0),
                               ena_intensity_sys_err=np.average(intensity_sys_err, axis=0, weights=exposures),
                               ena_intensity_stat_unc=combined_intensity_stat_unc
                               )
