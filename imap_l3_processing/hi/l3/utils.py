from pathlib import Path
from typing import Union

from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.maps.map_models import HiGlowsL3eData, HiL1cData


def read_hi_l1c_data(path: Union[Path, str]) -> HiL1cData:
    with CDF(str(path)) as cdf:
        return HiL1cData(epoch=cdf["epoch"][0], epoch_j2000=cdf.raw_var("epoch")[...],
                         exposure_times=read_numeric_variable(cdf["exposure_times"]),
                         esa_energy_step=cdf["esa_energy_step"][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> HiGlowsL3eData:
    with CDF(str(cdf_path)) as cdf:
        return HiGlowsL3eData(epoch=cdf["epoch"][0],
                              energy=read_numeric_variable(cdf["energy"]),
                              spin_angle=read_numeric_variable(cdf["spin_angle"]),
                              probability_of_survival=read_numeric_variable(cdf["probability_of_survival"]))
