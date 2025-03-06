from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf_file_path: Union[str, Path]) -> HitL2Data:
    with CDF(str(cdf_file_path)) as cdf:
        return HitL2Data(
            epoch=cdf.raw_var("epoch")[...],
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
            cno_energy=cdf["cno_energy"][...],
            cno_energy_delta_plus=cdf["cno_energy_delta_plus"][...],
            cno_energy_delta_minus=cdf["cno_energy_delta_minus"][...],
            fe_energy=cdf["fe_energy"][...],
            fe_energy_delta_plus=cdf["fe_energy_delta_plus"][...],
            fe_energy_delta_minus=cdf["fe_energy_delta_minus"][...],
            h_energy=cdf["h_energy"][...],
            h_energy_delta_plus=cdf["h_energy_delta_plus"][...],
            h_energy_delta_minus=cdf["h_energy_delta_minus"][...],
            he4_energy=cdf["he4_energy"][...],
            he4_energy_delta_plus=cdf["he4_energy_delta_plus"][...],
            he4_energy_delta_minus=cdf["he4_energy_delta_minus"][...],
            nemgsi_energy=cdf["nemgsi_energy"][...],
            nemgsi_energy_delta_plus=cdf["nemgsi_energy_delta_plus"][...],
            nemgsi_energy_delta_minus=cdf["nemgsi_energy_delta_minus"][...],
        )
