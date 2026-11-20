from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.models import DataProduct, DataProductVariable

ELECTRON_EPOCH_DELTA_NS = 30 * 1_000_000_000


@dataclass
class HitL1bIalirtRates:
    epoch: np.ndarray
    rates: np.ndarray
    rates_delta_plus: np.ndarray
    rates_delta_minus: np.ndarray

    @classmethod
    def read_from_cdf(cls, cdf_file_path: Path | str) -> "HitL1bIalirtRates":
        with CDF(str(cdf_file_path)) as cdf:
            rates = read_numeric_variable(cdf["ialirtrates"])
            return cls(
                epoch=cdf["epoch"][...],
                rates=rates,
                rates_delta_plus=np.where(
                    np.isnan(rates),
                    np.nan,
                    cdf["ialirtrates_stat_uncert_plus"][...].astype(float),
                ),
                rates_delta_minus=np.where(
                    np.isnan(rates),
                    np.nan,
                    cdf["ialirtrates_stat_uncert_minus"][...].astype(float),
                ),
            )


@dataclass
class HitElectronDataProduct(DataProduct):
    epoch: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    energy_labels: list[str]
    a_side: np.ndarray
    a_side_delta_plus: np.ndarray
    a_side_delta_minus: np.ndarray
    b_side: np.ndarray
    b_side_delta_plus: np.ndarray
    b_side_delta_minus: np.ndarray
    hit_flags: np.ndarray

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable("epoch", self.epoch),
            DataProductVariable("epoch_delta", np.full(len(self.epoch), ELECTRON_EPOCH_DELTA_NS)),
            DataProductVariable("electron_energy", self.energy),
            DataProductVariable("electron_energy_delta_plus", self.energy_delta_plus),
            DataProductVariable("electron_energy_delta_minus", self.energy_delta_minus),
            DataProductVariable("electron_energy_label", self.energy_labels),
            DataProductVariable("electron_intensity_a_side", self.a_side),
            DataProductVariable("electron_intensity_a_side_delta_plus", self.a_side_delta_plus),
            DataProductVariable("electron_intensity_a_side_delta_minus", self.a_side_delta_minus),
            DataProductVariable("electron_intensity_b_side", self.b_side),
            DataProductVariable("electron_intensity_b_side_delta_plus", self.b_side_delta_plus),
            DataProductVariable("electron_intensity_b_side_delta_minus", self.b_side_delta_minus),
            DataProductVariable("hit_flags", self.hit_flags),
        ]
