from dataclasses import dataclass
from datetime import datetime
from typing import Self

import numpy as np

from imap_l3_processing.constants import CARRINGTON_ROTATION_IN_NANOSECONDS
from imap_l3_processing.glows.l3bc.l3bc_toolkit.l3b_CarringtonIonRate import CarringtonIonizationRate
from imap_l3_processing.models import DataProduct, DataProductVariable, UpstreamDataDependency


@dataclass
class CRToProcess:
    l3a_paths: list[str]
    cr_midpoint: str
    cr_rotation_number: int


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

    @classmethod
    def from_instrument_team_object(cls, model: CarringtonIonizationRate,
                                    input_metadata: UpstreamDataDependency) -> Self:
        latitude_grid = model.carr_ion_rate["ion_grid"]
        return cls(
            input_metadata=input_metadata,
            epoch=np.array([model.carr_ion_rate["date"]]),
            epoch_delta=np.array([CARRINGTON_ROTATION_IN_NANOSECONDS / 2]),
            cr=np.array([model.carr_ion_rate["CR"]]),
            uv_anisotropy_factor=np.array([model.uv_anisotropy]),
            lat_grid=np.array(latitude_grid),
            lat_grid_delta=np.zeros(len(latitude_grid)),
            sum_rate=np.array([model.carr_ion_rate["ion_rate"]]),
            ph_rate=np.array([model.carr_ion_rate["ph_rate"]]),
            cx_rate=np.array([model.carr_ion_rate["cx_rate"]]),
            sum_uncert=np.array([model.carr_ion_rate["ion_rate_uncert"]]),
            ph_uncert=np.array([model.carr_ion_rate["ph_rate_uncert"]]),
            cx_uncert=np.array([model.carr_ion_rate["cx_rate_uncert"]]),
            lat_grid_label=[f"{x}°" for x in latitude_grid],
        )
