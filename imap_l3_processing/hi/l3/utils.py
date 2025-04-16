from pathlib import Path
from typing import Union

from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_float_variable
from imap_l3_processing.hi.l3.models import HiMapData, HiL1cData, GlowsL3eData, HiIntensityMapData


def read_hi_l2_data(cdf_path) -> HiIntensityMapData:
    with CDF(str(cdf_path)) as cdf:
        return HiIntensityMapData(
            epoch=read_float_variable(cdf["Epoch"]),
            epoch_delta=cdf["epoch_delta"][...],
            energy=read_float_variable(cdf["energy"]),
            energy_delta_plus=read_float_variable(cdf["energy_delta_plus"]),
            energy_delta_minus=read_float_variable(cdf["energy_delta_minus"]),
            energy_label=cdf["energy_label"][...],
            latitude=read_float_variable(cdf["latitude"]),
            latitude_delta=read_float_variable(cdf["latitude_delta"]),
            latitude_label=cdf["latitude_label"][...],
            longitude=read_float_variable(cdf["longitude"]),
            longitude_delta=read_float_variable(cdf["longitude_delta"]),
            longitude_label=cdf["longitude_label"][...],
            exposure_factor=read_float_variable(cdf["exposure_factor"]),
            obs_date=read_float_variable(cdf["obs_date"]),
            obs_date_range=cdf["obs_date_range"][...],
            solid_angle=read_float_variable(cdf["solid_angle"]),
            ena_intensity=read_float_variable(cdf["ena_intensity"]),
            ena_intensity_stat_unc=read_float_variable(cdf["ena_intensity_stat_unc"]),
            ena_intensity_sys_err=read_float_variable(cdf["ena_intensity_sys_err"]),
        )


def read_hi_l1c_data(path: Union[Path, str]) -> HiL1cData:
    with CDF(str(path)) as cdf:
        return HiL1cData(epoch=cdf["epoch"][0], epoch_j2000=cdf.raw_var("epoch")[...],
                         exposure_times=read_float_variable(cdf["exposure_times"]),
                         esa_energy_step=cdf["esa_energy_step"][...])


def read_glows_l3e_data(cdf_path: Union[Path, str]) -> GlowsL3eData:
    with CDF(str(cdf_path)) as cdf:
        return GlowsL3eData(epoch=cdf["epoch"][0],
                            energy=read_float_variable(cdf["energy"]),
                            spin_angle=read_float_variable(cdf["spin_angle"]),
                            probability_of_survival=read_float_variable(cdf["probability_of_survival"]))
