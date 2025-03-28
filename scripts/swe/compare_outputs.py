from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyhdf
from pyhdf.HDF import *
from pyhdf.VS import *
from spacepy.pycdf import CDF

_ = pyhdf
_ = pyhdf.VS
_ = VS

variable_mapping = {
    "spacecraft_potential": "potential",
    "core_halo_breakpoint": "break_core_halo",
    "core_chisq": "chisq_c",
    "halo_chisq": "chisq_h",
    "core_density_fit": "n_fc",
    "halo_density_fit": "n_fh",
    "core_t_parallel_fit": "t_para_fc",
    "halo_t_parallel_fit": "t_para_fh",
    "core_t_perpendicular_fit": "t_perp_fc",
    "halo_t_perpendicular_fit": "t_perp_fh",
    "core_temperature_phi_rtn_fit": "t_phi_fc",
    "halo_temperature_phi_rtn_fit": "t_phi_fh",
    "core_temperature_theta_rtn_fit": "t_theta_fc",
    "halo_temperature_theta_rtn_fit": "t_theta_fh",
    "core_speed_fit": "v_fc",
    "halo_speed_fit": "v_fh",
    "core_velocity_vector_rtn_fit": "v_rtn_fc",
    "halo_velocity_vector_rtn_fit": "v_rtn_fh",
    "core_density_integrated": "n_ic",
    "halo_density_integrated": "n_ih",
    "total_density_integrated": "n_i",
    "core_speed_integrated": "v_ic",
    "halo_speed_integrated": "v_ih",
    "total_speed_integrated": "v_i",
    "core_velocity_vector_rtn_integrated": "v_rtn_ic",
    "halo_velocity_vector_rtn_integrated": "v_rtn_ih",
    "total_velocity_vector_rtn_integrated": "v_rtn_i",
    "core_heat_flux_magnitude_integrated": "q_flux_ic",
    "core_heat_flux_theta_integrated": "q_flux_theta_ic",
    "core_heat_flux_phi_integrated": "q_flux_phi_ic",
    "halo_heat_flux_magnitude_integrated": "q_flux_ih",
    "halo_heat_flux_theta_integrated": "q_flux_theta_ih",
    "halo_heat_flux_phi_integrated": "q_flux_phi_ih",
    "total_heat_flux_magnitude_integrated": "q_flux_i",
    "total_heat_flux_theta_integrated": "q_flux_theta_i",
    "total_heat_flux_phi_integrated": "q_flux_phi_i",
    "core_t_parallel_integrated": "t_para_ic",
    "core_t_perpendicular_integrated": "t_perp_ic",
    "halo_t_parallel_integrated": "t_para_ih",
    "halo_t_perpendicular_integrated": "t_perp_ih",
    "total_t_parallel_integrated": "t_para_i",
    "total_t_perpendicular_integrated": "t_perp_i",
    "core_temperature_theta_rtn_integrated": "t_theta_ic",
    "core_temperature_phi_rtn_integrated": "t_phi_ic",
    "halo_temperature_theta_rtn_integrated": "t_theta_ih",
    "halo_temperature_phi_rtn_integrated": "t_phi_ih",
    "total_temperature_theta_rtn_integrated": "t_theta_i",
    "total_temperature_phi_rtn_integrated": "t_phi_i",
    "core_temperature_parallel_to_mag": "tc_para_b",
    "core_temperature_perpendicular_to_mag": "tc_perp_b",
    "halo_temperature_parallel_to_mag": "th_para_b",
    "halo_temperature_perpendicular_to_mag": "th_perp_b",
    "total_temperature_parallel_to_mag": "t_para_b",
    "total_temperature_perpendicular_to_mag": "t_perp_b",
}
Path("comparisons").mkdir(exist_ok=True)
for modern_name, heritage_name in variable_mapping.items():
    l3_hdf = HDF(r'instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf')
    l3_vs = l3_hdf.vstart()
    l3_swe_e = l3_vs.attach("swepam_e")
    heritage_data = [x[l3_swe_e.field(heritage_name)._index] for x in l3_swe_e[:]]

    l3_output_cdf = CDF('temp_cdf_data/imap_swe_l3_sci_20250630_v000.cdf')

    modern_data = l3_output_cdf[modern_name][:]
    nand_data = np.where((modern_data == -1e31), np.nan, modern_data)
    plt.plot([i for i in range(0, len(heritage_data))], heritage_data, label=f'Heritage: {heritage_name}')
    plt.plot([i for i in range(0, len(modern_data))], nand_data, label=f'Modern: {modern_name}')

    plt.legend()
    plt.savefig(Path("comparisons") / f"{modern_name}.png")
    plt.clf()
