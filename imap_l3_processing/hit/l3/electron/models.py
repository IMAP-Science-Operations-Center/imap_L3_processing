from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable


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
