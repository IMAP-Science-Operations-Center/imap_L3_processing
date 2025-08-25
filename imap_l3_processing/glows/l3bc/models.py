import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Self, Optional

import imap_data_access
import numpy as np
from astropy.time import Time
from spacepy import pycdf

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import jd_fm_Carrington
from imap_l3_processing.glows.l3bc.utils import get_date_range_of_cr, get_best_ancillary
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata


@dataclass
class L3BCAncillaryQueryResults:
    uv_anisotropy: list[dict]
    waw_helio_ion_mp: list[dict]
    bad_days_list: list[dict]
    pipeline_settings: list[dict]

    @classmethod
    def fetch(cls):
        return cls(
            uv_anisotropy=imap_data_access.query(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR"),
            waw_helio_ion_mp=imap_data_access.query(table="ancillary", instrument="glows", descriptor="WawHelioIonMP"),
            bad_days_list=imap_data_access.query(table="ancillary", instrument="glows", descriptor="bad-days-list"),
            pipeline_settings=imap_data_access.query(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde"),
        )

    def missing_ancillaries(self) -> bool:
        for ancillary_files in dataclasses.asdict(self).values():
            if len(ancillary_files) == 0:
                return True
        return False


@dataclass
class CRToProcess:
    l3a_file_names: set[str]
    uv_anisotropy_file_name: str
    waw_helio_ion_mp_file_name: str
    bad_days_list_file_name: str
    pipeline_settings_file_name: str

    cr_start_date: datetime
    cr_end_date: datetime
    cr_rotation_number: int

    def pipeline_dependency_file_names(self) -> set[str]:
        ancillary_files = {
            self.uv_anisotropy_file_name,
            self.waw_helio_ion_mp_file_name,
            self.bad_days_list_file_name,
            self.pipeline_settings_file_name
        }

        return self.l3a_file_names | ancillary_files

    def buffer_time_has_elapsed_since_cr(self):
        pipeline_settings_local_path = imap_data_access.download(self.pipeline_settings_file_name)
        pipeline_settings = read_pipeline_settings(pipeline_settings_local_path)
        buffer_time = timedelta(days=pipeline_settings["initializer_time_delta_days"])
        return datetime.now() > buffer_time + self.cr_end_date


def read_pipeline_settings(pipeline_settings_file_path: Path) -> dict:
    with open(pipeline_settings_file_path) as pipeline_settings:
        pipeline_settings = json.load(pipeline_settings)
    return pipeline_settings

@dataclass
class GlowsL3BIonizationRate(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[float]
    cr: np.ndarray[float]
    uv_anisotropy_factor: np.ndarray[float]
    lat_grid: np.ndarray[float]
    lat_grid_delta: np.ndarray[float]
    sum_rate: np.ndarray[float]
    ph_rate: np.ndarray[float]
    cx_rate: np.ndarray[float]
    sum_uncert: np.ndarray[float]
    ph_uncert: np.ndarray[float]
    cx_uncert: np.ndarray[float]
    lat_grid_label: list[str]
    mean_time: np.ndarray[datetime]
    uv_anisotropy_flag: np.ndarray[int]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [DataProductVariable("epoch", self.epoch),
                DataProductVariable("epoch_delta", self.epoch_delta),
                DataProductVariable("cr", self.cr),
                DataProductVariable("uv_anisotropy_factor", self.uv_anisotropy_factor),
                DataProductVariable("lat_grid", self.lat_grid),
                DataProductVariable("lat_grid_delta", self.lat_grid_delta),
                DataProductVariable("sum_rate", self.sum_rate),
                DataProductVariable("ph_rate", self.ph_rate),
                DataProductVariable("cx_rate", self.cx_rate),
                DataProductVariable("sum_uncert", self.sum_uncert),
                DataProductVariable("ph_uncert", self.ph_uncert),
                DataProductVariable("cx_uncert", self.cx_uncert),
                DataProductVariable("lat_grid_label", self.lat_grid_label),
                DataProductVariable("mean_time", self.mean_time),
                DataProductVariable("uv_anisotropy_flag", self.uv_anisotropy_flag),
                ]

    @classmethod
    def from_instrument_team_dictionary(cls, model: dict, input_metadata: InputMetadata) -> Self:
        latitude_grid = model["ion_rate_profile"]["lat_grid"]
        carrington_center_point = Time(jd_fm_Carrington(model["CR"] + 0.5), format="jd").datetime
        parent_file_names = []
        parent_file_names += collect_file_names(model['header']['ancillary_data_files'])
        parent_file_names += collect_file_names(model['header']['external_dependeciens'])
        parent_file_names += collect_file_names(model['header']['l3a_input_files_name'])
        return cls(
            input_metadata=input_metadata,
            parent_file_names=parent_file_names,
            epoch=np.array([carrington_center_point]),
            epoch_delta=np.array([CARRINGTON_ROTATION_IN_NANOSECONDS / 2]),
            cr=np.array([model["CR"]]),
            uv_anisotropy_factor=np.array([model["uv_anisotropy_factor"]]),
            lat_grid=np.array(latitude_grid),
            lat_grid_delta=np.zeros(len(latitude_grid)),
            sum_rate=np.array([model["ion_rate_profile"]["sum_rate"]]),
            ph_rate=np.array([model["ion_rate_profile"]["ph_rate"]]),
            cx_rate=np.array([model["ion_rate_profile"]["cx_rate"]]),
            sum_uncert=np.array([model["ion_rate_profile"]["sum_uncert"]]),
            ph_uncert=np.array([model["ion_rate_profile"]["ph_uncert"]]),
            cx_uncert=np.array([model["ion_rate_profile"]["cx_uncert"]]),
            lat_grid_label=[f"{x}°" for x in latitude_grid],
            mean_time=np.array([datetime.fromisoformat(model["date"])]),
            uv_anisotropy_flag=np.array([model['uv_anisotropy_flag']])
        )


@dataclass
class GlowsL3CSolarWind(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[float]
    cr: np.ndarray[float]
    lat_grid: np.ndarray[float]
    lat_grid_delta: np.ndarray[float]
    lat_grid_label: list[str]
    plasma_speed_ecliptic: np.ndarray[float]
    proton_density_ecliptic: np.ndarray[float]
    alpha_abundance_ecliptic: np.ndarray[float]
    plasma_speed_profile: np.ndarray[float]
    proton_density_profile: np.ndarray[float]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("epoch", self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable("epoch_delta", self.epoch_delta, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable("cr", self.cr, cdf_data_type=pycdf.const.CDF_INT2),
            DataProductVariable("lat_grid", self.lat_grid, cdf_data_type=pycdf.const.CDF_FLOAT, record_varying=False),
            DataProductVariable("lat_grid_delta", self.lat_grid_delta, cdf_data_type=pycdf.const.CDF_FLOAT,
                                record_varying=False),
            DataProductVariable("lat_grid_label", self.lat_grid_label, cdf_data_type=pycdf.const.CDF_CHAR,
                                record_varying=False),
            DataProductVariable("plasma_speed_ecliptic", self.plasma_speed_ecliptic,
                                cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable("proton_density_ecliptic", self.proton_density_ecliptic,
                                cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable("alpha_abundance_ecliptic", self.alpha_abundance_ecliptic,
                                cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable("plasma_speed_profile", self.plasma_speed_profile, cdf_data_type=pycdf.const.CDF_FLOAT),
            DataProductVariable("proton_density_profile", self.proton_density_profile,
                                cdf_data_type=pycdf.const.CDF_FLOAT),
        ]

    @classmethod
    def from_instrument_team_dictionary(cls, model: dict, input_metadata: InputMetadata) -> Self:
        latitude_grid = model["solar_wind_profile"]["lat_grid"]
        carrington_center_point = Time(jd_fm_Carrington(model["CR"] + 0.5), format="jd").datetime
        parent_file_names = []
        parent_file_names += collect_file_names(model['header']['ancillary_data_files'])
        parent_file_names += collect_file_names(model['header']['external_dependeciens'])
        return cls(
            input_metadata=input_metadata,
            epoch=np.array([carrington_center_point]),
            epoch_delta=np.array([CARRINGTON_ROTATION_IN_NANOSECONDS / 2]),
            cr=np.array([model['CR']]),
            lat_grid=np.array(latitude_grid),
            lat_grid_delta=np.zeros(len(latitude_grid)),
            lat_grid_label=[f"{x}°" for x in latitude_grid],
            plasma_speed_ecliptic=np.array([model["solar_wind_ecliptic"]['plasma_speed']]),
            proton_density_ecliptic=np.array([model["solar_wind_ecliptic"]['proton_density']]),
            alpha_abundance_ecliptic=np.array([model["solar_wind_ecliptic"]['alpha_abundance']]),
            plasma_speed_profile=np.array([model["solar_wind_profile"]['plasma_speed']]),
            proton_density_profile=np.array([model["solar_wind_profile"]['proton_density']]),
            parent_file_names=parent_file_names,
        )


def collect_values(source: dict | list | str) -> Iterable[str]:
    if isinstance(source, dict):
        return collect_values(list(source.values()))
    elif isinstance(source, list):
        return chain.from_iterable([collect_values(value) for value in source])
    else:
        return [source]


def collect_file_names(source: dict | list | str) -> list[str]:
    return [Path(file).name for file in collect_values(source)]
