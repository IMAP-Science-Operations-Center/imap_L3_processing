from pathlib import Path
from unittest.mock import sentinel

import numpy as np
import spiceypy
from imap_data_access import SPICEFilePath

from imap_l3_processing.swe.l3.science.pitch_calculations import calculate_velocity_in_dsp_frame_km_s, \
    swe_rebin_intensity_by_pitch_angle_and_gyrophase, correct_and_rebin
from imap_l3_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_l3_processing.swe.swe_processor import SweProcessor

swe_l2_file_path = Path(r'data/imap/swe/l2/2025/12/imap_swe_l2_sci_20251217_v002.cdf')
swe_l1b_file_path = Path(r'data/imap/swe/l1b/2025/12/imap_swe_l1b_sci_20251217_v004.cdf')
mag_file_path = Path(r'data/imap/mag/l1d/2025/12/imap_mag_l1d_norm-dsrf_20251217_v001.cdf')
swapi_file_path = Path(r'data/imap/swapi/l3a/2025/12/imap_swapi_l3a_proton-sw_20251217_v002.cdf')
configuration_file_path = Path(r'data/imap/ancillary/swe/imap_swe_config_20251119_v002.json')

spice_files = [
    "naif0012.tls",
    "pck00011.tpc",
    "imap_130.tf",
    "imap_science_110.tf",
    "imap_sclk_0072.tsc",
    "de440.bsp",
    "imap_pred_od013_20251201_20260112_v01.bsp",
    "imap_dps_2025_350_2025_351_001.ah.bc",
    "imap_dps_2025_350_2025_352_001.ah.bc",
    "imap_dps_2025_351_2025_353_001.ah.bc"
]

for spice in spice_files:
    spiceypy.furnsh(str(SPICEFilePath(spice).construct_path()))

swe_l3_dependencies = SweL3Dependencies.from_file_paths(swe_l2_file_path, swe_l1b_file_path, mag_file_path,
                                                        swapi_file_path, configuration_file_path)

swe_processor = SweProcessor(sentinel.pic, sentinel.input_metadata)
corrected_energy_bins = np.array([[-3.2314079545247543, -2.1864079545247543, -0.7614079545247536,
                                   1.281092045475246, 4.083592045475245, 7.978592045475245, 13.393592045475241,
                                   20.898592045475244, 31.396092045475246, 45.978592045475246,
                                   66.26109204547525, 94.47609204547524, 133.71109204547523, 188.28859204547524,
                                   264.2410920454752, 369.8335920454752, 516.7510920454753, 721.0960920454753,
                                   1005.3360920454752, 1400.7735920454752, 1950.7760920454753,
                                   2715.811092045475, 3780.001092045475, 5260.243592045475]])
# swe_processor.calculate_pitch_angle_products(swe_l3_dependencies, corrected_energy_bins)
swe_l2_data = swe_l3_dependencies.swe_l2_data
counts = swe_l3_dependencies.swe_l1b_data.count_rates * (swe_l2_data.acquisition_duration[..., np.newaxis] / 1e6)
rebinned_mag_data = np.array([-0.91321456, 0.96044633, 6.76249351])[np.newaxis, np.newaxis, np.newaxis, :]


def calculate(i):
    dsp_velocities = calculate_velocity_in_dsp_frame_km_s(corrected_energy_bins[i], swe_l2_data.inst_el,
                                                          swe_l2_data.inst_az_spin_sector[i])

    rebinned_psd, rebinned_psd_by_pa_and_gyro = correct_and_rebin(swe_l2_data.phase_space_density[i],
                                                                  np.array([0, 0, -400]),
                                                                  dsp_velocities,
                                                                  rebinned_mag_data[i],
                                                                  swe_l3_dependencies.configuration)
    swe_rebin_intensity_by_pitch_angle_and_gyrophase(
        swe_l2_data.flux[i],
        counts[i],
        dsp_velocities,
        rebinned_mag_data,
        swe_l3_dependencies.configuration)


calculate(0)
