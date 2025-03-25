from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from spacepy import pycdf

from imap_l3_processing.models import DataProduct, DataProductVariable

EPOCH_CDF_VAR_NAME = "epoch"
EPOCH_DELTA_CDF_VAR_NAME = "epoch_delta"
ENERGY_CDF_VAR_NAME = "energy"
ENERGY_DELTA_PLUS_CDF_VAR_NAME = "energy_delta_plus"
ENERGY_DELTA_MINUS_CDF_VAR_NAME = "energy_delta_minus"
PITCH_ANGLE_CDF_VAR_NAME = "pitch_angle"
PITCH_ANGLE_DELTA_CDF_VAR_NAME = "pitch_angle_delta"
GYROPHASE_BINS_CDF_VAR_NAME = "gyrophase_bins"
GYROPHASE_DELTA_CDF_VAR_NAME = "gyrophase_deltas"
PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME = "phase_space_density_by_pitch_angle"
PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME = "phase_space_density_by_pitch_angle_and_gyrophase"
ENERGY_SPECTRUM_CDF_VAR_NAME = "energy_spectrum"
ENERGY_SPECTRUM_INBOUND_CDF_VAR_NAME = "energy_spectrum_inbound"
ENERGY_SPECTRUM_OUTBOUND_CDF_VAR_NAME = "energy_spectrum_outbound"
INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME = "intensity_by_pitch_angle"
INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME = "intensity_by_pitch_angle_and_gyrophase"
INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_CDF_VAR_NAME = "intensity_uncertainty_by_pitch_angle"
INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME = "intensity_uncertainty_by_pitch_angle_and_gyrophase"
SPACECRAFT_POTENTIAL_CDF_VAR_NAME = "spacecraft_potential"
CORE_HALO_BREAKPOINT_CDF_VAR_NAME = "core_halo_breakpoint"
CORE_FIT_NUM_POINTS_CDF_VAR_NAME = "core_fit_num_points"
CHISQ_C_CDF_VAR_NAME = "core_chisq"
CHISQ_H_CDF_VAR_NAME = "halo_chisq"
CORE_DENSITY_FIT_CDF_VAR_NAME = "core_density_fit"
HALO_DENSITY_FIT_CDF_VAR_NAME = "halo_density_fit"
CORE_T_PARALLEL_FIT_CDF_VAR_NAME = "core_t_parallel_fit"
HALO_T_PARALLEL_FIT_CDF_VAR_NAME = "halo_t_parallel_fit"
CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME = "core_t_perpendicular_fit"
HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME = "halo_t_perpendicular_fit"
CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME = "core_temperature_phi_rtn_fit"
HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME = "halo_temperature_phi_rtn_fit"
CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME = "core_temperature_theta_rtn_fit"
HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME = "halo_temperature_theta_rtn_fit"
CORE_SPEED_FIT_CDF_VAR_NAME = "core_speed_fit"
HALO_SPEED_FIT_CDF_VAR_NAME = "halo_speed_fit"
CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME = "core_velocity_vector_rtn_fit"
HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME = "halo_velocity_vector_rtn_fit"
CORE_DENSITY_INTEGRATED_CDF_VAR_NAME = "core_density_integrated"
HALO_DENSITY_INTEGRATED_CDF_VAR_NAME = "halo_density_integrated"
TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME = "total_density_integrated"
CORE_SPEED_INTEGRATED_CDF_VAR_NAME = "core_speed_integrated"
HALO_SPEED_INTEGRATED_CDF_VAR_NAME = "halo_speed_integrated"
TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME = "total_speed_integrated"
CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME = "core_velocity_vector_rtn_integrated"
HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME = "halo_velocity_vector_rtn_integrated"
TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME = "total_velocity_vector_rtn_integrated"
CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME = "core_heat_flux_magnitude_integrated"
CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME = "core_heat_flux_theta_integrated"
CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME = "core_heat_flux_phi_integrated"
HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME = "halo_heat_flux_magnitude_integrated"
HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME = "halo_heat_flux_theta_integrated"
HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME = "halo_heat_flux_phi_integrated"
TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME = "total_heat_flux_magnitude_integrated"
TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME = "total_heat_flux_theta_integrated"
TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME = "total_heat_flux_phi_integrated"
CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME = "core_t_parallel_integrated"
CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME = "core_t_perpendicular_integrated"
HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME = "halo_t_parallel_integrated"
HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME = "halo_t_perpendicular_integrated"
TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME = "total_t_parallel_integrated"
TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME = "total_t_perpendicular_integrated"
CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME = "core_temperature_theta_rtn_integrated"
CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME = "core_temperature_phi_rtn_integrated"
HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME = "halo_temperature_theta_rtn_integrated"
HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME = "halo_temperature_phi_rtn_integrated"
TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME = "total_temperature_theta_rtn_integrated"
TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME = "total_temperature_phi_rtn_integrated"
CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME = "core_temperature_parallel_to_mag"
CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME = "core_temperature_perpendicular_to_mag"
HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME = "halo_temperature_parallel_to_mag"
HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME = "halo_temperature_perpendicular_to_mag"
TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME = "total_temperature_parallel_to_mag"
TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME = "total_temperature_perpendicular_to_mag"
CORE_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME = "core_temperature_tensor_integrated"
HALO_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME = "halo_temperature_tensor_integrated"
TOTAL_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME = "total_temperature_tensor_integrated"


@dataclass
class SweL2Data:
    epoch: np.ndarray
    phase_space_density: np.ndarray
    flux: np.ndarray  # actually flux_spin_sector
    energy: np.ndarray
    inst_el: np.ndarray
    inst_az_spin_sector: np.ndarray
    acquisition_time: np.ndarray
    acquisition_duration: np.ndarray


@dataclass
class SwapiL3aProtonData:
    epoch: np.ndarray
    epoch_delta: np.ndarray
    proton_sw_speed: np.ndarray[float]
    proton_sw_clock_angle: np.ndarray[float]
    proton_sw_deflection_angle: np.ndarray[float]


@dataclass
class SweL1bData:
    epoch: np.ndarray
    count_rates: np.ndarray
    settle_duration: np.ndarray


@dataclass
class SweL3MomentData:
    core_fit_num_points: np.ndarray
    core_chisq: np.ndarray
    halo_chisq: np.ndarray
    core_density_fit: np.ndarray
    halo_density_fit: np.ndarray
    core_t_parallel_fit: np.ndarray
    halo_t_parallel_fit: np.ndarray
    core_t_perpendicular_fit: np.ndarray
    halo_t_perpendicular_fit: np.ndarray
    core_temperature_phi_rtn_fit: np.ndarray
    halo_temperature_phi_rtn_fit: np.ndarray
    core_temperature_theta_rtn_fit: np.ndarray
    halo_temperature_theta_rtn_fit: np.ndarray
    core_speed_fit: np.ndarray
    halo_speed_fit: np.ndarray
    core_velocity_vector_rtn_fit: np.ndarray
    halo_velocity_vector_rtn_fit: np.ndarray
    core_density_integrated: np.ndarray
    halo_density_integrated: np.ndarray
    total_density_integrated: np.ndarray
    core_speed_integrated: np.ndarray
    halo_speed_integrated: np.ndarray
    total_speed_integrated: np.ndarray
    core_velocity_vector_rtn_integrated: np.ndarray
    halo_velocity_vector_rtn_integrated: np.ndarray
    total_velocity_vector_rtn_integrated: np.ndarray
    core_heat_flux_magnitude_integrated: np.ndarray
    core_heat_flux_theta_integrated: np.ndarray
    core_heat_flux_phi_integrated: np.ndarray
    halo_heat_flux_magnitude_integrated: np.ndarray
    halo_heat_flux_theta_integrated: np.ndarray
    halo_heat_flux_phi_integrated: np.ndarray
    total_heat_flux_magnitude_integrated: np.ndarray
    total_heat_flux_theta_integrated: np.ndarray
    total_heat_flux_phi_integrated: np.ndarray
    core_t_parallel_integrated: np.ndarray
    core_t_perpendicular_integrated: np.ndarray
    halo_t_parallel_integrated: np.ndarray
    halo_t_perpendicular_integrated: np.ndarray
    total_t_parallel_integrated: np.ndarray
    total_t_perpendicular_integrated: np.ndarray
    core_temperature_theta_rtn_integrated: np.ndarray
    core_temperature_phi_rtn_integrated: np.ndarray
    halo_temperature_theta_rtn_integrated: np.ndarray
    halo_temperature_phi_rtn_integrated: np.ndarray
    total_temperature_theta_rtn_integrated: np.ndarray
    total_temperature_phi_rtn_integrated: np.ndarray
    core_temperature_parallel_to_mag: np.ndarray
    core_temperature_perpendicular_to_mag: np.ndarray
    halo_temperature_parallel_to_mag: np.ndarray
    halo_temperature_perpendicular_to_mag: np.ndarray
    total_temperature_parallel_to_mag: np.ndarray
    total_temperature_perpendicular_to_mag: np.ndarray
    core_temperature_tensor_integrated: np.ndarray
    halo_temperature_tensor_integrated: np.ndarray
    total_temperature_tensor_integrated: np.ndarray


@dataclass
class SweL3Data(DataProduct):
    # coming from the config
    epoch: np.ndarray
    epoch_delta: np.ndarray
    energy: np.ndarray
    energy_delta_plus: np.ndarray
    energy_delta_minus: np.ndarray
    pitch_angle: np.ndarray
    pitch_angle_delta: np.ndarray
    gyrophase_bins: np.ndarray
    gyrophase_delta: np.ndarray
    spacecraft_potential: np.ndarray
    core_halo_breakpoint: np.ndarray
    # intensity
    intensity_by_pitch_angle: np.ndarray
    intensity_by_pitch_angle_and_gyrophase: np.ndarray
    intensity_uncertainty_by_pitch_angle: np.ndarray
    intensity_uncertainty_by_pitch_angle_and_gyrophase: np.ndarray
    # pitch angle specific
    phase_space_density_by_pitch_angle: np.ndarray
    phase_space_density_by_pitch_angle_and_gyrophase: np.ndarray
    energy_spectrum: np.ndarray
    energy_spectrum_inbound: np.ndarray
    energy_spectrum_outbound: np.ndarray
    # fit moments
    moment_data: SweL3MomentData

    def to_data_product_variables(self) -> list[DataProductVariable]:
        return [
            DataProductVariable(EPOCH_CDF_VAR_NAME, value=self.epoch, cdf_data_type=pycdf.const.CDF_TIME_TT2000),
            DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, value=self.epoch_delta,
                                cdf_data_type=pycdf.const.CDF_INT8),
            DataProductVariable(ENERGY_CDF_VAR_NAME, value=self.energy, cdf_data_type=pycdf.const.CDF_REAL4,
                                record_varying=False),
            DataProductVariable(ENERGY_DELTA_PLUS_CDF_VAR_NAME, value=self.energy_delta_plus,
                                cdf_data_type=pycdf.const.CDF_REAL4, record_varying=False),
            DataProductVariable(ENERGY_DELTA_MINUS_CDF_VAR_NAME, value=self.energy_delta_minus,
                                cdf_data_type=pycdf.const.CDF_REAL4, record_varying=False),
            DataProductVariable(PITCH_ANGLE_CDF_VAR_NAME, value=self.pitch_angle, cdf_data_type=pycdf.const.CDF_REAL4,
                                record_varying=False),
            DataProductVariable(PITCH_ANGLE_DELTA_CDF_VAR_NAME, value=self.pitch_angle_delta,
                                cdf_data_type=pycdf.const.CDF_REAL4, record_varying=False),

            DataProductVariable(GYROPHASE_BINS_CDF_VAR_NAME, value=self.gyrophase_bins,
                                cdf_data_type=pycdf.const.CDF_REAL4, record_varying=False),
            DataProductVariable(GYROPHASE_DELTA_CDF_VAR_NAME, value=self.gyrophase_delta,
                                cdf_data_type=pycdf.const.CDF_REAL4, record_varying=False),
            DataProductVariable(PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME,
                                value=self.phase_space_density_by_pitch_angle, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME,
                                value=self.phase_space_density_by_pitch_angle_and_gyrophase,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(ENERGY_SPECTRUM_CDF_VAR_NAME,
                                value=self.energy_spectrum, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(ENERGY_SPECTRUM_INBOUND_CDF_VAR_NAME,
                                value=self.energy_spectrum_inbound, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(ENERGY_SPECTRUM_OUTBOUND_CDF_VAR_NAME,
                                value=self.energy_spectrum_outbound, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME,
                                value=self.intensity_by_pitch_angle, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME,
                                value=self.intensity_by_pitch_angle_and_gyrophase, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_CDF_VAR_NAME,
                                value=self.intensity_uncertainty_by_pitch_angle, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(INTENSITY_UNCERTAINTY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME,
                                value=self.intensity_uncertainty_by_pitch_angle_and_gyrophase,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(SPACECRAFT_POTENTIAL_CDF_VAR_NAME,
                                value=self.spacecraft_potential, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_HALO_BREAKPOINT_CDF_VAR_NAME,
                                value=self.core_halo_breakpoint, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_FIT_NUM_POINTS_CDF_VAR_NAME,
                                value=self.moment_data.core_fit_num_points, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CHISQ_C_CDF_VAR_NAME,
                                value=self.moment_data.core_chisq, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CHISQ_H_CDF_VAR_NAME,
                                value=self.moment_data.halo_chisq, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_DENSITY_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_density_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_DENSITY_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_density_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_T_PARALLEL_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_t_parallel_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_T_PARALLEL_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_t_parallel_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_t_perpendicular_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_t_perpendicular_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_phi_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_phi_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_theta_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_theta_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_SPEED_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_speed_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_SPEED_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_speed_fit, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.core_velocity_vector_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME,
                                value=self.moment_data.halo_velocity_vector_rtn_fit,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_DENSITY_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_density_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_DENSITY_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_density_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_density_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_SPEED_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_speed_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_SPEED_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_speed_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_speed_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_velocity_vector_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_velocity_vector_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_velocity_vector_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_heat_flux_magnitude_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_heat_flux_theta_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_heat_flux_phi_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_heat_flux_magnitude_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_heat_flux_theta_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_heat_flux_phi_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_heat_flux_magnitude_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_heat_flux_theta_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_heat_flux_phi_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_t_parallel_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_t_perpendicular_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_t_parallel_integrated, cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_t_perpendicular_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_t_parallel_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_t_perpendicular_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_theta_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_phi_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_theta_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_phi_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_temperature_theta_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_temperature_phi_rtn_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_parallel_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_perpendicular_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_parallel_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_perpendicular_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.total_temperature_parallel_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME,
                                value=self.moment_data.total_temperature_perpendicular_to_mag,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(CORE_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.core_temperature_tensor_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(HALO_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.halo_temperature_tensor_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
            DataProductVariable(TOTAL_TEMPERATURE_TENSOR_INTEGRATED_CDF_VAR_NAME,
                                value=self.moment_data.total_temperature_tensor_integrated,
                                cdf_data_type=pycdf.const.CDF_REAL4),
        ]


class SweConfiguration(TypedDict):
    geometric_fractions: list[float]
    pitch_angle_bins: list[float]
    pitch_angle_delta: list[float]
    energy_bins: list[float]
    energy_delta_plus: list[float]
    energy_delta_minus: list[float]
    energy_bin_low_multiplier: float
    energy_bin_high_multiplier: float
    in_vs_out_energy_index: float
    high_energy_proximity_threshold: float
    low_energy_proximity_threshold: float
    max_swapi_offset_in_minutes: float
    max_mag_offset_in_minutes: float
    slope_ratio_cutoff_for_potential_calc: float
    spacecraft_potential_initial_guess: float
    core_halo_breakpoint_initial_guess: float
    core_energy_for_slope_guess: float
    halo_energy_for_slope_guess: float
    refit_core_halo_breakpoint_index: int
    minimum_phase_space_density_value: float
    aperture_field_of_view_radians: list[float]
