from pathlib import Path
from typing import Union

from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable
from imap_l3_processing.hi.l3.models import HiMapData, HiL1cData, GlowsL3eData


def read_hi_l2_data(cdf_path) -> HiMapData:
    with CDF(str(cdf_path)) as cdf:
        return HiMapData(
            epoch=cdf["Epoch"][...],
            energy=cdf["bin"][...],
            energy_deltas=cdf["bin_boundaries"][...],
            counts=read_variable(cdf["counts"]),
            counts_uncertainty=read_variable(cdf["counts_uncertainty"]),
            epoch_delta=cdf["epoch_delta"][...],
            exposure=read_variable(cdf["exposure"]),
            flux=read_variable(cdf["flux"]),
            lat=cdf["lat"][...],
            lon=cdf["lon"][...],
            sensitivity=read_variable(cdf["sensitivity"]),
            variance=read_variable(cdf["variance"])
        )


def read_hi_l1c_data(path: Union[Path, str]) -> HiL1cData:
    with CDF(str(path)) as cdf:
        return HiL1cData(epoch=cdf["epoch"][0], epoch_j2000=cdf.raw_var("epoch")[...],
                         exposure_times=read_variable(cdf["exposure_times"]),
                         esa_energy_step=cdf["esa_energy_step"][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> GlowsL3eData:
    with CDF(str(cdf_path)) as cdf:
        return GlowsL3eData(epoch=cdf["epoch"][0],
                            energy=read_variable(cdf["energy"]),
                            spin_angle=read_variable(cdf["spin_angle"]),
                            probability_of_survival=read_variable(cdf["probability_of_survival"]))
