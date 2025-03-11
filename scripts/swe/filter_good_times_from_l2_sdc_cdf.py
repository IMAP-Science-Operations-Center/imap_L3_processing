import sys

import numpy as np
from spacepy.pycdf import CDF

bad_time_cdf = CDF(sys.argv[1])

with CDF("imap_swe_l2_sci-fake_20240510_v002.cdf", masterpath="", readonly=False) as w:
    w["epoch"] = np.array([bad_time_cdf["epoch"][2], bad_time_cdf["epoch"][4]])
    w["epoch"].attrs = bad_time_cdf["epoch"].attrs
    w["acq_duration"] = np.array([bad_time_cdf["acq_duration"][2], bad_time_cdf["acq_duration"][4]])
    w["acq_duration"].attrs = bad_time_cdf["acq_duration"].attrs
    w["acquisition_time"] = np.array([bad_time_cdf["acquisition_time"][2], bad_time_cdf["acquisition_time"][4]])
    w["acquisition_time"].attrs = bad_time_cdf["acquisition_time"].attrs
    w["flux_spin_sector"] = np.array([bad_time_cdf["flux_spin_sector"][2], bad_time_cdf["flux_spin_sector"][4]])
    w["flux_spin_sector"].attrs = bad_time_cdf["flux_spin_sector"].attrs
    w["inst_az_spin_sector"] = np.array(
        [bad_time_cdf["inst_az_spin_sector"][2], bad_time_cdf["inst_az_spin_sector"][4]])
    w["inst_az_spin_sector"].attrs = bad_time_cdf["inst_az_spin_sector"].attrs
    w["phase_space_density"] = np.array(
        [bad_time_cdf["phase_space_density"][2], bad_time_cdf["phase_space_density"][4]])
    w["phase_space_density"].attrs = bad_time_cdf["phase_space_density"].attrs
    w["phase_space_density_spin_sector"] = np.array(
        [bad_time_cdf["phase_space_density_spin_sector"][2], bad_time_cdf["phase_space_density_spin_sector"][4]])
    w["phase_space_density_spin_sector"].attrs = bad_time_cdf["phase_space_density_spin_sector"].attrs
    w["flux"] = np.array([bad_time_cdf["flux"][2], bad_time_cdf["flux"][4]])
    w["flux"].attrs = bad_time_cdf["flux"].attrs

    w["cem_id"] = bad_time_cdf["cem_id"]
    w["cem_id"].attrs = bad_time_cdf["cem_id"].attrs
    w["cem_id_label"] = bad_time_cdf["cem_id_label"]
    w["cem_id_label"].attrs = bad_time_cdf["cem_id_label"].attrs
    w["energy"] = bad_time_cdf["energy"]
    w["energy"].attrs = bad_time_cdf["energy"].attrs
    w["energy_label"] = bad_time_cdf["energy_label"]
    w["energy_label"].attrs = bad_time_cdf["energy_label"].attrs
    w["esa_step"] = bad_time_cdf["esa_step"]
    w["esa_step"].attrs = bad_time_cdf["esa_step"].attrs
    w["esa_step_label"] = bad_time_cdf["esa_step_label"]
    w["esa_step_label"].attrs = bad_time_cdf["esa_step_label"].attrs
    w["inst_az"] = bad_time_cdf["inst_az"]
    w["inst_az"].attrs = bad_time_cdf["inst_az"].attrs
    w["inst_az_label"] = bad_time_cdf["inst_az_label"]
    w["inst_az_label"].attrs = bad_time_cdf["inst_az_label"].attrs
    w["inst_el"] = bad_time_cdf["inst_el"]
    w["inst_el"].attrs = bad_time_cdf["inst_el"].attrs
    w["inst_el_label"] = bad_time_cdf["inst_el_label"]
    w["inst_el_label"].attrs = bad_time_cdf["inst_el_label"].attrs
    w["spin_sector"] = bad_time_cdf["spin_sector"]
    w["spin_sector"].attrs = bad_time_cdf["spin_sector"].attrs
    w["spin_sector_label"] = bad_time_cdf["spin_sector_label"]
    w["spin_sector_label"].attrs = bad_time_cdf["spin_sector_label"].attrs
