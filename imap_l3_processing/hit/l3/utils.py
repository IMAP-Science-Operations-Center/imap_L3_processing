from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_float_variable
from imap_l3_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf_file_path: Union[str, Path]) -> HitL2Data:
    with CDF(str(cdf_file_path)) as cdf:
        return HitL2Data(
            epoch=cdf["epoch"][...],
            epoch_delta=np.array([timedelta(seconds=ns / 1e9) for ns in cdf["epoch_delta"][...]]),
            h=read_float_variable(cdf["h"]),
            he4=read_float_variable(cdf["he4"]),
            cno=read_float_variable(cdf["cno"]),
            nemgsi=read_float_variable(cdf["nemgsi"]),
            fe=read_float_variable(cdf["fe"]),
            delta_minus_cno=read_float_variable(cdf["delta_minus_cno"]),
            delta_minus_he4=read_float_variable(cdf["delta_minus_he4"]),
            delta_minus_h=read_float_variable(cdf["delta_minus_h"]),
            delta_minus_fe=read_float_variable(cdf["delta_minus_fe"]),
            delta_minus_nemgsi=read_float_variable(cdf["delta_minus_nemgsi"]),
            delta_plus_cno=read_float_variable(cdf["delta_plus_cno"]),
            delta_plus_he4=read_float_variable(cdf["delta_plus_he4"]),
            delta_plus_h=read_float_variable(cdf["delta_plus_h"]),
            delta_plus_fe=read_float_variable(cdf["delta_plus_fe"]),
            delta_plus_nemgsi=read_float_variable(cdf["delta_plus_nemgsi"]),
            cno_energy=cdf["cno_energy_mean"][...],
            cno_energy_delta_plus=cdf["cno_energy_delta_plus"][...],
            cno_energy_delta_minus=cdf["cno_energy_delta_minus"][...],
            fe_energy=cdf["fe_energy_mean"][...],
            fe_energy_delta_plus=cdf["fe_energy_delta_plus"][...],
            fe_energy_delta_minus=cdf["fe_energy_delta_minus"][...],
            h_energy=cdf["h_energy_mean"][...],
            h_energy_delta_plus=cdf["h_energy_delta_plus"][...],
            h_energy_delta_minus=cdf["h_energy_delta_minus"][...],
            he4_energy=cdf["he4_energy_mean"][...],
            he4_energy_delta_plus=cdf["he4_energy_delta_plus"][...],
            he4_energy_delta_minus=cdf["he4_energy_delta_minus"][...],
            nemgsi_energy=cdf["nemgsi_energy_mean"][...],
            nemgsi_energy_delta_plus=cdf["nemgsi_energy_delta_plus"][...],
            nemgsi_energy_delta_minus=cdf["nemgsi_energy_delta_minus"][...],
        )
