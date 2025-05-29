from pathlib import Path
from typing import Union

from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import PROBABILITY_OF_SURVIVAL_VAR_NAME, EPOCH_CDF_VAR_NAME, \
    ENERGY_VAR_NAME, SPIN_ANGLE_VAR_NAME
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet


def read_l1c_rectangular_pointing_set_data(path: Union[Path, str]) -> InputRectangularPointingSet:
    with CDF(str(path)) as cdf:
        return InputRectangularPointingSet(epoch=cdf["epoch"][0], epoch_j2000=cdf.raw_var("epoch")[...],
                                           exposure_times=read_numeric_variable(cdf["exposure_times"]),
                                           esa_energy_step=cdf["esa_energy_step"][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> GlowsL3eRectangularMapInputData:
    with CDF(str(cdf_path)) as cdf:
        return GlowsL3eRectangularMapInputData(epoch=cdf[EPOCH_CDF_VAR_NAME][0],
                                               energy=read_numeric_variable(cdf[ENERGY_VAR_NAME]),
                                               spin_angle=read_numeric_variable(cdf[SPIN_ANGLE_VAR_NAME]),
                                               probability_of_survival=read_numeric_variable(
                                                   cdf[PROBABILITY_OF_SURVIVAL_VAR_NAME]))
