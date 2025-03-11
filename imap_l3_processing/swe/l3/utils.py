import json
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable
from imap_l3_processing.swe.l3.models import SweConfiguration, SweL2Data, SwapiL3aProtonData

from imap_processing.spice.time import met_to_datetime64


def read_l2_swe_data(swe_l2_data: Path) -> SweL2Data:
    with CDF(str(swe_l2_data)) as cdf:
        epoch = cdf["epoch"][:]
        flux = read_variable(cdf["flux_spin_sector"])
        inst_el = cdf["inst_el"][:]
        energy = cdf["energy"][:]
        inst_az_spin_sector = read_variable(cdf["inst_az_spin_sector"])
        phase_space_density = read_variable(cdf["phase_space_density_spin_sector"])
        acquisition_time_in_MET = read_variable(cdf["acquisition_time"])
        valid_times_mask = np.isfinite(acquisition_time_in_MET)
        converted_valid_times = met_to_datetime64(acquisition_time_in_MET[valid_times_mask].ravel())
        acquisition_time_dt64 = np.full(acquisition_time_in_MET.shape, np.datetime64("NaT", 'ns'))
        acquisition_time_dt64[valid_times_mask] = converted_valid_times
    return SweL2Data(epoch=epoch,
                     epoch_delta=np.full(epoch.shape, timedelta(seconds=30)),
                     phase_space_density=phase_space_density,
                     flux=flux,
                     energy=energy,
                     inst_el=inst_el,
                     inst_az_spin_sector=inst_az_spin_sector,
                     acquisition_time=acquisition_time_dt64,
                     )


def read_l3a_swapi_proton_data(swapi_l3a_data: Path) -> SwapiL3aProtonData:
    with CDF(str(swapi_l3a_data)) as cdf:
        epoch = cdf["epoch"][:]
        epoch_delta_ns = cdf["epoch_delta"][...] / 1e9
        if cdf["epoch_delta"].rv():
            epoch_delta = [timedelta(seconds=x) for x in epoch_delta_ns]
        else:
            epoch_delta = np.repeat(timedelta(seconds=epoch_delta_ns), len(epoch))
        proton_sw_speed = read_variable(cdf["proton_sw_speed"])
        proton_sw_clock_angle = read_variable(cdf["proton_sw_clock_angle"])
        proton_sw_deflection_angle = read_variable(cdf["proton_sw_deflection_angle"])

    return SwapiL3aProtonData(epoch=epoch,
                              epoch_delta=epoch_delta,
                              proton_sw_speed=proton_sw_speed,
                              proton_sw_clock_angle=proton_sw_clock_angle,
                              proton_sw_deflection_angle=proton_sw_deflection_angle)


def read_swe_config(swe_config: Path) -> SweConfiguration:
    with swe_config.open() as f:
        config: SweConfiguration = json.load(f)

    return config
