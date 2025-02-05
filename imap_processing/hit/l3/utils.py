import math
from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf_file_path: Union[str, Path]) -> HitL2Data:
    with CDF(str(cdf_file_path)) as cdf:
        return HitL2Data(
            epoch=cdf["epoch"][...],
            epoch_delta=np.array([timedelta(seconds=ns / 1e9) for ns in cdf["epoch_delta"][...]]),
            hydrogen=cdf["hydrogen"][...],
            helium4=cdf["helium4"][...],
            CNO=cdf["CNO"][...],
            NeMgSi=cdf["NeMgSi"][...],
            iron=cdf["iron"][...],
            DELTA_MINUS_CNO=cdf["DELTA_MINUS_CNO"][...],
            DELTA_MINUS_HELIUM4=cdf["DELTA_MINUS_HELIUM4"][...],
            DELTA_MINUS_HYDROGEN=cdf["DELTA_MINUS_HYDROGEN"][...],
            DELTA_MINUS_IRON=cdf["DELTA_MINUS_IRON"][...],
            DELTA_MINUS_NEMGSI=cdf["DELTA_MINUS_NEMGSI"][...],
            DELTA_PLUS_CNO=cdf["DELTA_PLUS_CNO"][...],
            DELTA_PLUS_HELIUM4=cdf["DELTA_PLUS_HELIUM4"][...],
            DELTA_PLUS_HYDROGEN=cdf["DELTA_PLUS_HYDROGEN"][...],
            DELTA_PLUS_IRON=cdf["DELTA_PLUS_IRON"][...],
            DELTA_PLUS_NEMGSI=cdf["DELTA_PLUS_NEMGSI"][...],
            cno_energy_high=cdf["cno_energy_high"][...],
            cno_energy_idx=cdf["cno_energy_idx"][...],
            cno_energy_low=cdf["cno_energy_low"][...],
            fe_energy_high=cdf["fe_energy_high"][...],
            fe_energy_idx=cdf["fe_energy_idx"][...],
            fe_energy_low=cdf["fe_energy_low"][...],
            h_energy_high=cdf["h_energy_high"][...],
            h_energy_idx=cdf["h_energy_idx"][...],
            h_energy_low=cdf["h_energy_low"][...],
            he4_energy_high=cdf["he4_energy_high"][...],
            he4_energy_idx=cdf["he4_energy_idx"][...],
            he4_energy_low=cdf["he4_energy_low"][...],
            nemgsi_energy_high=cdf["nemgsi_energy_high"][...],
            nemgsi_energy_idx=cdf["nemgsi_energy_idx"][...],
            nemgsi_energy_low=cdf["nemgsi_energy_low"][...],
        )
