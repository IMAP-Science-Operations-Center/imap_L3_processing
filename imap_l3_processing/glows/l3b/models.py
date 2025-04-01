from dataclasses import dataclass
from datetime import datetime

import numpy as np

from imap_l3_processing.models import DataProduct, DataProductVariable


@dataclass
class CRToProcess:
    l3a_paths: list[str]
    cr_midpoint: str
    cr_rotation_number: int
    uv_anisotropy: str
    waw_helioion_mp: str


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
                DataProductVariable("lat_grid_label", self.lat_grid_label)]
