import sys

import numpy as np
from spacepy.pycdf import CDF

l1b_cdf = CDF(sys.argv[1])

with CDF("imap_swe_l1b_sci-3-11-good-times_20100101_v002", masterpath="", readonly=False) as cdf:
    cdf["acq_duration"] = np.array([l1b_cdf["acq_duration"][2], l1b_cdf["acq_duration"][4]])
    cdf["acq_duration"].attrs = l1b_cdf["acq_duration"].attrs
    cdf["acq_start_coarse"] = np.array([l1b_cdf["acq_start_coarse"][2], l1b_cdf["acq_start_coarse"][4]])
    cdf["acq_start_coarse"].attrs = l1b_cdf["acq_start_coarse"].attrs
    cdf["acq_start_fine"] = np.array([l1b_cdf["acq_start_fine"][2], l1b_cdf["acq_start_fine"][4]])
    cdf["acq_start_fine"].attrs = l1b_cdf["acq_start_fine"].attrs
    cdf["acquisition_time"] = np.array([l1b_cdf["acquisition_time"][2], l1b_cdf["acquisition_time"][4]])
    cdf["acquisition_time"].attrs = l1b_cdf["acquisition_time"].attrs
    cdf["cem_nominal_only"] = np.array([l1b_cdf["cem_nominal_only"][2], l1b_cdf["cem_nominal_only"][4]])
    cdf["cem_nominal_only"].attrs = l1b_cdf["cem_nominal_only"].attrs
    cdf["cksum"] = np.array([l1b_cdf["cksum"][2], l1b_cdf["cksum"][4]])
    cdf["cksum"].attrs = l1b_cdf["cksum"].attrs
    cdf["epoch"] = np.array([l1b_cdf["epoch"][2], l1b_cdf["epoch"][4]])
    cdf["epoch"].attrs = l1b_cdf["epoch"].attrs
    cdf["esa_acq_cfg"] = np.array([l1b_cdf["esa_acq_cfg"][2], l1b_cdf["esa_acq_cfg"][4]])
    cdf["esa_acq_cfg"].attrs = l1b_cdf["esa_acq_cfg"].attrs
    cdf["esa_table_num"] = np.array([l1b_cdf["esa_table_num"][2], l1b_cdf["esa_table_num"][4]])
    cdf["esa_table_num"].attrs = l1b_cdf["esa_table_num"].attrs
    cdf["high_count"] = np.array([l1b_cdf["high_count"][2], l1b_cdf["high_count"][4]])
    cdf["high_count"].attrs = l1b_cdf["high_count"].attrs
    cdf["quarter_cycle"] = np.array([l1b_cdf["quarter_cycle"][2], l1b_cdf["quarter_cycle"][4]])
    cdf["quarter_cycle"].attrs = l1b_cdf["quarter_cycle"].attrs
    cdf["repoint_warning"] = np.array([l1b_cdf["repoint_warning"][2], l1b_cdf["repoint_warning"][4]])
    cdf["repoint_warning"].attrs = l1b_cdf["repoint_warning"].attrs
    cdf["science_data"] = np.array([l1b_cdf["science_data"][2], l1b_cdf["science_data"][4]])
    cdf["science_data"].attrs = l1b_cdf["science_data"].attrs
    cdf["settle_duration"] = np.array([l1b_cdf["settle_duration"][2], l1b_cdf["settle_duration"][4]])
    cdf["settle_duration"].attrs = l1b_cdf["settle_duration"].attrs
    cdf["shcoarse"] = np.array([l1b_cdf["shcoarse"][2], l1b_cdf["shcoarse"][4]])
    cdf["shcoarse"].attrs = l1b_cdf["shcoarse"].attrs
    cdf["spin_period"] = np.array([l1b_cdf["spin_period"][2], l1b_cdf["spin_period"][4]])
    cdf["spin_period"].attrs = l1b_cdf["spin_period"].attrs
    cdf["spin_period_source"] = np.array([l1b_cdf["spin_period_source"][2], l1b_cdf["spin_period_source"][4]])
    cdf["spin_period_source"].attrs = l1b_cdf["spin_period_source"].attrs
    cdf["spin_period_validity"] = np.array([l1b_cdf["spin_period_validity"][2], l1b_cdf["spin_period_validity"][4]])
    cdf["spin_period_validity"].attrs = l1b_cdf["spin_period_validity"].attrs
    cdf["spin_phase"] = np.array([l1b_cdf["spin_phase"][2], l1b_cdf["spin_phase"][4]])
    cdf["spin_phase"].attrs = l1b_cdf["spin_phase"].attrs
    cdf["spin_phase_validity"] = np.array([l1b_cdf["spin_phase_validity"][2], l1b_cdf["spin_phase_validity"][4]])
    cdf["spin_phase_validity"].attrs = l1b_cdf["spin_phase_validity"].attrs
    cdf["stim_cfg_reg"] = np.array([l1b_cdf["stim_cfg_reg"][2], l1b_cdf["stim_cfg_reg"][4]])
    cdf["stim_cfg_reg"].attrs = l1b_cdf["stim_cfg_reg"].attrs
    cdf["stim_enabled"] = np.array([l1b_cdf["stim_enabled"][2], l1b_cdf["stim_enabled"][4]])
    cdf["stim_enabled"].attrs = l1b_cdf["stim_enabled"].attrs
    cdf["threshold_dac"] = np.array([l1b_cdf["threshold_dac"][2], l1b_cdf["threshold_dac"][4]])
    cdf["threshold_dac"].attrs = l1b_cdf["threshold_dac"].attrs

    cdf["cycle"] = l1b_cdf["cycle"]
    cdf["cycle"].attrs = l1b_cdf["cycle"].attrs
    cdf["cem_id"] = l1b_cdf["cem_id"]
    cdf["cem_id"].attrs = l1b_cdf["cem_id"].attrs
    cdf["esa_step_label"] = l1b_cdf["esa_step_label"]
    cdf["esa_step_label"].attrs = l1b_cdf["esa_step_label"].attrs
    cdf["spin_sector_label"] = l1b_cdf["spin_sector_label"]
    cdf["spin_sector_label"].attrs = l1b_cdf["spin_sector_label"].attrs
    cdf["cem_id_label"] = l1b_cdf["cem_id_label"]
    cdf["cem_id_label"].attrs = l1b_cdf["cem_id_label"].attrs
    cdf["spin_sector"] = l1b_cdf["spin_sector"]
    cdf["spin_sector"].attrs = l1b_cdf["spin_sector"].attrs
    cdf["esa_step"] = l1b_cdf["esa_step"]
    cdf["esa_step"].attrs = l1b_cdf["esa_step"].attrs
