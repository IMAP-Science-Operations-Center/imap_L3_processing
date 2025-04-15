from dataclasses import dataclass
from pathlib import Path

from numpy import ndarray
from spacepy.pycdf import CDF


@dataclass
class CodiceLoL2Data:
    epoch: ndarray
    epoch_delta: ndarray
    energy: ndarray
    spin_sector: ndarray
    ssd_id: ndarray
    h_intensities: ndarray
    he_intensities: ndarray
    c4_intensities: ndarray
    c5_intensities: ndarray
    c6_intensities: ndarray
    o5_intensities: ndarray
    o6_intensities: ndarray
    o7_intensities: ndarray
    o8_intensities: ndarray
    mg_intensities: ndarray
    si_intensities: ndarray
    fe_low_intensities: ndarray
    fe_high_intensities: ndarray

    @classmethod
    def read_from_cdf(cls, l2_sectored_intensities_cdf: Path):
        with CDF(str(l2_sectored_intensities_cdf)) as cdf:
            return cls(
                epoch=cdf["epoch"][...],
                epoch_delta=cdf["epoch_delta"][...],
                energy=cdf["energy"][...],
                spin_sector=cdf["spin_sector"][...],
                ssd_id=cdf["ssd_id"][...],
                h_intensities=cdf["h_intensities"][...],
                he_intensities=cdf["he_intensities"][...],
                c4_intensities=cdf["c4_intensities"][...],
                c5_intensities=cdf["c5_intensities"][...],
                c6_intensities=cdf["c6_intensities"][...],
                o5_intensities=cdf["o5_intensities"][...],
                o6_intensities=cdf["o6_intensities"][...],
                o7_intensities=cdf["o7_intensities"][...],
                o8_intensities=cdf["o8_intensities"][...],
                mg_intensities=cdf["mg_intensities"][...],
                si_intensities=cdf["si_intensities"][...],
                fe_low_intensities=cdf["fe_low_intensities"][...],
                fe_high_intensities=cdf["fe_high_intensities"][...],
            )
