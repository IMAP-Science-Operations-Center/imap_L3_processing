from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Optional

import imap_data_access
import numpy as np
from spacepy import pycdf

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.dependency_validator import validate_dependencies
from imap_l3_processing.glows.l3bc.utils import get_date_range_of_cr
from imap_l3_processing.models import DataProduct, DataProductVariable, InputMetadata
from imap_l3_processing.utils import download_external_dependency

F107_FLUX_TABLE_URL = "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
LYMAN_ALPHA_COMPOSITE_INDEX_URL = "https://lasp.colorado.edu/data/timed_see/composite_lya/lyman_alpha_composite.nc"
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"


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
            pipeline_settings=imap_data_access.query(table="ancillary", instrument="glows",
                                                     descriptor="pipeline-settings-l3bcde"),
        )

    def missing_ancillaries(self) -> bool:
        for ancillary_files in dataclasses.asdict(self).values():
            if len(ancillary_files) == 0:
                return True
        return False


@dataclass
class ExternalDependencies:
    f107_index_file_path: Path | None
    lyman_alpha_path: Path | None
    omni2_data_path: Path | None

    @classmethod
    def fetch_dependencies(cls):
        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL,
                                                            TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL,
                                                        TEMP_CDF_FOLDER_PATH / 'lyman_alpha.txt')
        omni2_data_path = download_external_dependency(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'omni2.txt')

        if f107_index_file_path is not None:
            ExternalDependencies._comment_headers(f107_index_file_path)

        return cls(f107_index_file_path, lyman_alpha_path, omni2_data_path)

    @staticmethod
    def _comment_headers(filename: Path, num_lines=2):
        with open(filename, "r+") as file:
            lines = file.readlines()
            for i in range(num_lines):
                lines[i] = "#" + lines[i]
            file.truncate(0)
        with open(filename, "w") as file:
            file.writelines(lines)


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

    _buffer_time: Optional[timedelta] = None

    def pipeline_dependency_file_names(self) -> set[str]:
        ancillary_files = {
            self.uv_anisotropy_file_name,
            self.waw_helio_ion_mp_file_name,
            self.bad_days_list_file_name,
            self.pipeline_settings_file_name
        }

        return self.l3a_file_names | ancillary_files

    def get_buffer_time(self) -> timedelta:
        if self._buffer_time is None:
            pipeline_settings_local_path = imap_data_access.download(self.pipeline_settings_file_name)
            pipeline_settings = read_pipeline_settings(pipeline_settings_local_path)
            self._buffer_time = timedelta(days=pipeline_settings["initializer_time_delta_days"])
        return self._buffer_time

    def has_valid_external_dependencies(self, external_deps: ExternalDependencies) -> bool:
        return validate_dependencies(
            self.cr_end_date,
            self.get_buffer_time(),
            external_deps.omni2_data_path,
            external_deps.f107_index_file_path,
            external_deps.lyman_alpha_path
        )

    def buffer_time_has_elapsed_since_cr(self):
        return datetime.now() > self.get_buffer_time() + self.cr_end_date


def read_pipeline_settings(pipeline_settings_file_path: Path) -> dict:
    with open(pipeline_settings_file_path) as pipeline_settings:
        pipeline_settings = json.load(pipeline_settings)
    return pipeline_settings


@dataclass
class GlowsL3BIonizationRate(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta_plus: np.ndarray[int]
    epoch_delta_minus: np.ndarray[int]
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
    uv_anisotropy_flag: np.ndarray[int]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [DataProductVariable("epoch", self.epoch),
                DataProductVariable("epoch_delta_plus", self.epoch_delta_plus),
                DataProductVariable("epoch_delta_minus", self.epoch_delta_minus),
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
                DataProductVariable("uv_anisotropy_flag", self.uv_anisotropy_flag),
                ]

    @classmethod
    def from_instrument_team_dictionary(cls, model: dict, input_metadata: InputMetadata) -> GlowsL3BIonizationRate:
        latitude_grid = model["ion_rate_profile"]["lat_grid"]
        mean_time = datetime.fromisoformat(model["date"])
        start_of_cr, end_of_cr = get_date_range_of_cr(model["CR"])

        epoch = mean_time
        epoch_delta_plus = (end_of_cr - mean_time).total_seconds() * 1e9
        epoch_delta_minus = (mean_time - start_of_cr).total_seconds() * 1e9

        parent_file_names = []
        parent_file_names += collect_file_names(model['header']['ancillary_data_files'])
        parent_file_names += collect_file_names(model['header']['external_dependeciens'])
        return cls(
            input_metadata=input_metadata,
            parent_file_names=parent_file_names,
            epoch=np.array([epoch]),
            epoch_delta_plus=np.array([epoch_delta_plus]),
            epoch_delta_minus=np.array([epoch_delta_minus]),
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
            uv_anisotropy_flag=np.array([model['uv_anisotropy_flag']])
        )


@dataclass
class GlowsL3BCProcessorOutput:
    l3bs_by_cr: dict[int, str]
    l3cs_by_cr: dict[int, str]
    data_products: list[Path]


@dataclass
class GlowsL3CSolarWind(DataProduct):
    epoch: np.ndarray[datetime]
    epoch_delta_plus: np.ndarray[int]
    epoch_delta_minus: np.ndarray[int]
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
            DataProductVariable("epoch_delta_plus", self.epoch_delta_plus, cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable("epoch_delta_minus", self.epoch_delta_minus, cdf_data_type=pycdf.const.CDF_INT8),
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
    def from_instrument_team_dictionary(cls, model: dict, input_metadata: InputMetadata) -> GlowsL3CSolarWind:
        latitude_grid = model["solar_wind_profile"]["lat_grid"]
        mean_time = datetime.fromisoformat(model["date"])
        start_of_cr, end_of_cr = get_date_range_of_cr(model["CR"])

        epoch = mean_time
        epoch_delta_plus = (end_of_cr - mean_time).total_seconds() * 1e9
        epoch_delta_minus = (mean_time - start_of_cr).total_seconds() * 1e9

        parent_file_names = []
        parent_file_names += collect_file_names(model['header']['ancillary_data_files'])
        parent_file_names += collect_file_names(model['header']['external_dependeciens'])
        return cls(
            input_metadata=input_metadata,
            epoch=np.array([epoch]),
            epoch_delta_plus=np.array([epoch_delta_plus]),
            epoch_delta_minus=np.array([epoch_delta_minus]),
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
