import json
from datetime import timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.swe.l3.models import SweConfiguration, SweL2Data, SwapiL3aProtonData


def read_l2_swe_data(swe_l2_data: Path) -> SweL2Data:
    with CDF(str(swe_l2_data)) as cdf:
        epoch = cdf["epoch"][:]
        flux = cdf["flux_spin_sector"][:]
        inst_el = cdf["inst_el"][:]
        energy = cdf["energy"][:]
        inst_az_spin_sector = cdf["inst_az_spin_sector"][:]
        phase_space_density = cdf["phase_space_density_spin_sector"][:]
        acquisition_time_in_MET = cdf["acquisition_time"][:]
        mission_epoch = np.datetime64("2010-01-01", 'ns')
        wip_leap_second_correction_revisit_with_spice = 3
        acquisition_time = mission_epoch + (
                    (acquisition_time_in_MET - wip_leap_second_correction_revisit_with_spice) * 1e9).astype(int)
    return SweL2Data(epoch=epoch,
                     epoch_delta=np.full(epoch.shape, timedelta(seconds=30)),
                     phase_space_density=phase_space_density,
                     flux=flux,
                     energy=energy,
                     inst_el=inst_el,
                     inst_az_spin_sector=inst_az_spin_sector,
                     acquisition_time=acquisition_time,
                     )


def read_l3a_swapi_proton_data(swapi_l3a_data: Path) -> SwapiL3aProtonData:
    with CDF(str(swapi_l3a_data)) as cdf:
        epoch = cdf["epoch"][:]
        epoch_delta_ns = cdf["epoch_delta"][...] / 1e9
        if cdf["epoch_delta"].rv():
            epoch_delta = [timedelta(seconds=x) for x in epoch_delta_ns]
        else:
            epoch_delta = np.repeat(timedelta(seconds=epoch_delta_ns), len(epoch))
        proton_sw_speed = cdf["proton_sw_speed"][:]
        proton_sw_clock_angle = cdf["proton_sw_clock_angle"][:]
        proton_sw_deflection_angle = cdf["proton_sw_deflection_angle"][:]

    return SwapiL3aProtonData(epoch=epoch,
                              epoch_delta=epoch_delta,
                              proton_sw_speed=proton_sw_speed,
                              proton_sw_clock_angle=proton_sw_clock_angle,
                              proton_sw_deflection_angle=proton_sw_deflection_angle)


def read_swe_config(swe_config: Path) -> SweConfiguration:
    with swe_config.open() as f:
        config: SweConfiguration = json.load(f)

    return config
