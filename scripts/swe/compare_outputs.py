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
    "core_t_perpendicular_integrated": ("t_perp_ic", 0),
    "core_t_ratio_perpendicular_integrated": ("t_perp_ic", 1),
    "halo_t_parallel_integrated": "t_para_ih",
    "halo_t_perpendicular_integrated": ("t_perp_ih", 0),
    "halo_t_ratio_perpendicular_integrated": ("t_perp_ih", 1),
    "total_t_parallel_integrated": "t_para_i",
    "total_t_perpendicular_integrated": ("t_perp_i", 0),
    "total_t_ratio_perpendicular_integrated": ("t_perp_i", 1),
    "core_temperature_theta_rtn_integrated": "t_theta_ic",
    "core_temperature_phi_rtn_integrated": "t_phi_ic",
    "halo_temperature_theta_rtn_integrated": "t_theta_ih",
    "halo_temperature_phi_rtn_integrated": "t_phi_ih",
    "total_temperature_theta_rtn_integrated": "t_theta_i",
    "total_temperature_phi_rtn_integrated": "t_phi_i",
    "core_temperature_parallel_to_mag": "tc_para_b",
    "core_temperature_perpendicular_to_mag": ("tc_perp_b", 0),
    "halo_temperature_parallel_to_mag": "th_para_b",
    "halo_temperature_perpendicular_to_mag": ("th_perp_b", 0),
    "total_temperature_parallel_to_mag": "t_para_b",
    "total_temperature_perpendicular_to_mag": ("t_perp_b", 0),
    "core_temperature_ratio_perpendicular_to_mag": ("tc_perp_b", 1),
    "halo_temperature_ratio_perpendicular_to_mag": ("th_perp_b", 1),
    "total_temperature_ratio_perpendicular_to_mag": ("t_perp_b", 1),
    "core_temperature_tensor_integrated": "t_mat_ic",
    "halo_temperature_tensor_integrated": "t_mat_ih",
    "total_temperature_tensor_integrated": "t_mat_i",
}
Path("comparisons").mkdir(exist_ok=True)
for modern_name, heritage_info in variable_mapping.items():

    l3_hdf = HDF(r'instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf')
    l3_vs = l3_hdf.vstart()
    l3_swe_e = l3_vs.attach("swepam_e")
    match heritage_info:
        case (heritage_name, index):
            heritage_data = np.array([x[l3_swe_e.field(heritage_name)._index][index] for x in l3_swe_e[:]])
        case heritage_name:
            heritage_data = np.array([x[l3_swe_e.field(heritage_info)._index] for x in l3_swe_e[:]])

    l3_output_cdf = CDF('temp_cdf_data/imap_swe_l3_sci_20250629_v000.cdf')

    modern_data = l3_output_cdf[modern_name][:]
    nand_data = np.where((modern_data == -1e31), np.nan, modern_data)
    if "tensor" in modern_name:
        for i in range(6):
            plt.plot(heritage_data[:, i], label=f'Heritage: {heritage_name} {i}')
            plt.plot(nand_data[:, i], label=f'Modern: {modern_name} {i}')

            plt.legend()
            plt.savefig(Path("comparisons") / f"{modern_name}_{i}.png")
            plt.clf()
    else:
        plt.plot(heritage_data, label=f'Heritage: {heritage_name}')
        plt.plot(nand_data, label=f'Modern: {modern_name}')

        plt.legend()
        plt.savefig(Path("comparisons") / f"{modern_name}.png")
        plt.clf()
