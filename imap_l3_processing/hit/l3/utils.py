from pathlib import Path
from typing import Union

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf_file_path: Union[str, Path]) -> HitL2Data:
    with CDF(str(cdf_file_path)) as cdf:
        epoch = cdf["epoch"][...]

        half_epoch_diff = np.diff(epoch) / 2
        assert np.all(half_epoch_diff == half_epoch_diff[0])
        fabricated_epoch_deltas = np.repeat(half_epoch_diff[0], len(epoch))

        return HitL2Data(
            epoch=cdf["epoch"][...],
            epoch_delta=fabricated_epoch_deltas,
            h=np.transpose(read_numeric_variable(cdf["h_macropixel_intensity"]), axes=(0, 1, 3, 2)),
            he4=np.transpose(read_numeric_variable(cdf["he4_macropixel_intensity"]), axes=(0, 1, 3, 2)),
            cno=np.transpose(read_numeric_variable(cdf["cno_macropixel_intensity"]), axes=(0, 1, 3, 2)),
            nemgsi=np.transpose(read_numeric_variable(cdf["nemgsi_macropixel_intensity"]), axes=(0, 1, 3, 2)),
            fe=np.transpose(read_numeric_variable(cdf["fe_macropixel_intensity"]), axes=(0, 1, 3, 2)),
            delta_minus_cno=np.transpose(read_numeric_variable(cdf["cno_total_uncert_minus"]), axes=(0, 1, 3, 2)),
            delta_minus_he4=np.transpose(read_numeric_variable(cdf["he4_total_uncert_minus"]), axes=(0, 1, 3, 2)),
            delta_minus_h=np.transpose(read_numeric_variable(cdf["h_total_uncert_minus"]), axes=(0, 1, 3, 2)),
            delta_minus_fe=np.transpose(read_numeric_variable(cdf["fe_total_uncert_minus"]), axes=(0, 1, 3, 2)),
            delta_minus_nemgsi=np.transpose(read_numeric_variable(cdf["nemgsi_total_uncert_minus"]), axes=(0, 1, 3, 2)),
            delta_plus_cno=np.transpose(read_numeric_variable(cdf["cno_total_uncert_plus"]), axes=(0, 1, 3, 2)),
            delta_plus_he4=np.transpose(read_numeric_variable(cdf["he4_total_uncert_plus"]), axes=(0, 1, 3, 2)),
            delta_plus_h=np.transpose(read_numeric_variable(cdf["h_total_uncert_plus"]), axes=(0, 1, 3, 2)),
            delta_plus_fe=np.transpose(read_numeric_variable(cdf["fe_total_uncert_plus"]), axes=(0, 1, 3, 2)),
            delta_plus_nemgsi=np.transpose(read_numeric_variable(cdf["nemgsi_total_uncert_plus"]), axes=(0, 1, 3, 2)),
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
