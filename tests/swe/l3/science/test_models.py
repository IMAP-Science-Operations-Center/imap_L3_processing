import unittest
from datetime import timedelta
from unittest.mock import Mock

import numpy as np
from spacepy import pycdf

from imap_l3_processing.swe.l3.models import SweL3Data, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, \
    ENERGY_CDF_VAR_NAME, \
    ENERGY_DELTA_PLUS_CDF_VAR_NAME, ENERGY_DELTA_MINUS_CDF_VAR_NAME, PITCH_ANGLE_CDF_VAR_NAME, \
    PITCH_ANGLE_DELTA_CDF_VAR_NAME, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, \
    ENERGY_SPECTRUM_CDF_VAR_NAME, ENERGY_SPECTRUM_OUTBOUND_CDF_VAR_NAME, ENERGY_SPECTRUM_INBOUND_CDF_VAR_NAME, \
    SPACECRAFT_POTENTIAL_CDF_VAR_NAME, CORE_HALO_BREAKPOINT_CDF_VAR_NAME, CORE_FIT_NUM_POINTS_CDF_VAR_NAME, \
    CHISQ_C_CDF_VAR_NAME, CHISQ_H_CDF_VAR_NAME, CORE_DENSITY_FIT_CDF_VAR_NAME, HALO_DENSITY_FIT_CDF_VAR_NAME, \
    CORE_T_PARALLEL_FIT_CDF_VAR_NAME, HALO_T_PARALLEL_FIT_CDF_VAR_NAME, CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME, \
    HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME, CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, \
    HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, CORE_SPEED_FIT_CDF_VAR_NAME, HALO_SPEED_FIT_CDF_VAR_NAME, \
    CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, \
    CORE_DENSITY_INTEGRATED_CDF_VAR_NAME, HALO_DENSITY_INTEGRATED_CDF_VAR_NAME, TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME, \
    CORE_SPEED_INTEGRATED_CDF_VAR_NAME, HALO_SPEED_INTEGRATED_CDF_VAR_NAME, TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME, \
    CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, \
    CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, \
    HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, \
    HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, \
    CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, \
    TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, SweL3MomentData, CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, \
    TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_data_to_product_variables(self):
        epoch = Mock()
        epoch_delta = [30_000_000_000, 30_000_000_000]
        energy = Mock()
        energy_delta_plus = Mock()
        energy_delta_minus = Mock()
        pitch_angle = Mock()
        pitch_angle_delta = Mock()
        psd_by_pitch_angle = Mock()
        energy_spectrum = Mock()
        energy_spectrum_inbound = Mock()
        energy_spectrum_outbound = Mock()
        spacecraft_potential = Mock()
        core_halo_breakpoint = Mock()
        core_fit_num_points = Mock()
        chisq_c = Mock()
        chisq_h = Mock()
        core_density_fit = Mock()
        halo_density_fit = Mock()
        core_t_parallel_fit = Mock()
        halo_t_parallel_fit = Mock()
        core_t_perpendicular_fit = Mock()
        halo_t_perpendicular_fit = Mock()
        core_temperature_phi_rtn_fit = Mock()
        halo_temperature_phi_rtn_fit = Mock()
        core_temperature_theta_rtn_fit = Mock()
        halo_temperature_theta_rtn_fit = Mock()
        core_speed_fit = Mock()
        halo_speed_fit = Mock()
        core_velocity_vector_rtn_fit = Mock()
        halo_velocity_vector_rtn_fit = Mock()
        core_density_integrated = Mock()
        halo_density_integrated = Mock()
        total_density_integrated = Mock()
        core_speed_integrated = Mock()
        halo_speed_integrated = Mock()
        total_speed_integrated = Mock()
        core_velocity_vector_rtn_integrated = Mock()
        halo_velocity_vector_rtn_integrated = Mock()
        total_velocity_vector_rtn_integrated = Mock()
        core_heat_flux_magnitude_integrated = Mock()
        core_heat_flux_theta_integrated = Mock()
        core_heat_flux_phi_integrated = Mock()
        halo_heat_flux_magnitude_integrated = Mock()
        halo_heat_flux_theta_integrated = Mock()
        halo_heat_flux_phi_integrated = Mock()
        total_heat_flux_magnitude_integrated = Mock()
        total_heat_flux_theta_integrated = Mock()
        total_heat_flux_phi_integrated = Mock()
        core_t_parallel_integrated = Mock()
        core_t_perpendicular_integrated = Mock()
        halo_t_parallel_integrated = Mock()
        halo_t_perpendicular_integrated = Mock()
        total_t_parallel_integrated = Mock()
        total_t_perpendicular_integrated = Mock()
        core_temperature_theta_rtn_integrated = Mock()
        core_temperature_phi_rtn_integrated = Mock()
        halo_temperature_theta_rtn_integrated = Mock()
        halo_temperature_phi_rtn_integrated = Mock()
        total_temperature_theta_rtn_integrated = Mock()
        total_temperature_phi_rtn_integrated = Mock()
        core_temperature_parallel_to_mag = Mock()
        core_temperature_perpendicular_to_mag = Mock()
        halo_temperature_parallel_to_mag = Mock()
        halo_temperature_perpendicular_to_mag = Mock()
        total_temperature_parallel_to_mag = Mock()
        total_temperature_perpendicular_to_mag = Mock()

        data = SweL3Data(epoch=epoch,
                         epoch_delta=epoch_delta,
                         energy=energy,
                         energy_delta_plus=energy_delta_plus,
                         energy_delta_minus=energy_delta_minus,
                         pitch_angle=pitch_angle,
                         pitch_angle_delta=pitch_angle_delta,
                         phase_space_density_by_pitch_angle=psd_by_pitch_angle,
                         energy_spectrum=energy_spectrum,
                         energy_spectrum_inbound=energy_spectrum_inbound,
                         energy_spectrum_outbound=energy_spectrum_outbound,
                         spacecraft_potential=spacecraft_potential,
                         core_halo_breakpoint=core_halo_breakpoint,
                         moment_data=SweL3MomentData(
                             core_fit_num_points=core_fit_num_points,
                             core_chisq=chisq_c,
                             halo_chisq=chisq_h,
                             core_density_fit=core_density_fit,
                             halo_density_fit=halo_density_fit,
                             core_t_parallel_fit=core_t_parallel_fit,
                             halo_t_parallel_fit=halo_t_parallel_fit,
                             core_t_perpendicular_fit=core_t_perpendicular_fit,
                             halo_t_perpendicular_fit=halo_t_perpendicular_fit,
                             core_temperature_phi_rtn_fit=core_temperature_phi_rtn_fit,
                             halo_temperature_phi_rtn_fit=halo_temperature_phi_rtn_fit,
                             core_temperature_theta_rtn_fit=core_temperature_theta_rtn_fit,
                             halo_temperature_theta_rtn_fit=halo_temperature_theta_rtn_fit,
                             core_speed_fit=core_speed_fit,
                             halo_speed_fit=halo_speed_fit,
                             core_velocity_vector_rtn_fit=core_velocity_vector_rtn_fit,
                             halo_velocity_vector_rtn_fit=halo_velocity_vector_rtn_fit,
                             core_density_integrated=core_density_integrated,
                             halo_density_integrated=halo_density_integrated,
                             total_density_integrated=total_density_integrated,
                             core_speed_integrated=core_speed_integrated,
                             halo_speed_integrated=halo_speed_integrated,
                             total_speed_integrated=total_speed_integrated,
                             core_velocity_vector_rtn_integrated=core_velocity_vector_rtn_integrated,
                             halo_velocity_vector_rtn_integrated=halo_velocity_vector_rtn_integrated,
                             total_velocity_vector_rtn_integrated=total_velocity_vector_rtn_integrated,
                             core_heat_flux_magnitude_integrated=core_heat_flux_magnitude_integrated,
                             core_heat_flux_theta_integrated=core_heat_flux_theta_integrated,
                             core_heat_flux_phi_integrated=core_heat_flux_phi_integrated,
                             halo_heat_flux_magnitude_integrated=halo_heat_flux_magnitude_integrated,
                             halo_heat_flux_theta_integrated=halo_heat_flux_theta_integrated,
                             halo_heat_flux_phi_integrated=halo_heat_flux_phi_integrated,
                             total_heat_flux_magnitude_integrated=total_heat_flux_magnitude_integrated,
                             total_heat_flux_theta_integrated=total_heat_flux_theta_integrated,
                             total_heat_flux_phi_integrated=total_heat_flux_phi_integrated,
                             core_t_parallel_integrated=core_t_parallel_integrated,
                             core_t_perpendicular_integrated=core_t_perpendicular_integrated,
                             halo_t_parallel_integrated=halo_t_parallel_integrated,
                             halo_t_perpendicular_integrated=halo_t_perpendicular_integrated,
                             total_t_parallel_integrated=total_t_parallel_integrated,
                             total_t_perpendicular_integrated=total_t_perpendicular_integrated,
                             core_temperature_theta_rtn_integrated=core_temperature_theta_rtn_integrated,
                             core_temperature_phi_rtn_integrated=core_temperature_phi_rtn_integrated,
                             halo_temperature_theta_rtn_integrated=halo_temperature_theta_rtn_integrated,
                             halo_temperature_phi_rtn_integrated=halo_temperature_phi_rtn_integrated,
                             total_temperature_theta_rtn_integrated=total_temperature_theta_rtn_integrated,
                             total_temperature_phi_rtn_integrated=total_temperature_phi_rtn_integrated,
                             core_temperature_parallel_to_mag=core_temperature_parallel_to_mag,
                             core_temperature_perpendicular_to_mag=core_temperature_perpendicular_to_mag,
                             halo_temperature_parallel_to_mag=halo_temperature_parallel_to_mag,
                             halo_temperature_perpendicular_to_mag=halo_temperature_perpendicular_to_mag,
                             total_temperature_parallel_to_mag=total_temperature_parallel_to_mag,
                             total_temperature_perpendicular_to_mag=total_temperature_perpendicular_to_mag,
                         ),
                         input_metadata=Mock())

        variables = data.to_data_product_variables()

        self.assertEqual(66, len(variables))

        variables = iter(variables)
        # @formatter:off
        self.assert_variable_attributes(
            next(variables), epoch, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(
            next(variables), [30_000_000_000, 30_000_000_000], EPOCH_DELTA_CDF_VAR_NAME, pycdf.const.CDF_INT8)
        self.assert_variable_attributes(
            next(variables), energy, ENERGY_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), energy_delta_plus, ENERGY_DELTA_PLUS_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), energy_delta_minus, ENERGY_DELTA_MINUS_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), pitch_angle, PITCH_ANGLE_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), pitch_angle_delta, PITCH_ANGLE_DELTA_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), psd_by_pitch_angle, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum, ENERGY_SPECTRUM_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum_inbound, ENERGY_SPECTRUM_INBOUND_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum_outbound, ENERGY_SPECTRUM_OUTBOUND_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), spacecraft_potential, SPACECRAFT_POTENTIAL_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_halo_breakpoint, CORE_HALO_BREAKPOINT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)

        self.assert_variable_attributes(
            next(variables), core_fit_num_points, CORE_FIT_NUM_POINTS_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), chisq_c, CHISQ_C_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), chisq_h, CHISQ_H_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_density_fit, CORE_DENSITY_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_density_fit, HALO_DENSITY_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_t_parallel_fit, CORE_T_PARALLEL_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_t_parallel_fit, HALO_T_PARALLEL_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_t_perpendicular_fit, CORE_T_PERPENDICULAR_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_t_perpendicular_fit, HALO_T_PERPENDICULAR_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_phi_rtn_fit, CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_phi_rtn_fit, HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_theta_rtn_fit, CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_theta_rtn_fit, HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_speed_fit, CORE_SPEED_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_speed_fit, HALO_SPEED_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_velocity_vector_rtn_fit, CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_velocity_vector_rtn_fit, HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_density_integrated, CORE_DENSITY_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_density_integrated, HALO_DENSITY_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_density_integrated, TOTAL_DENSITY_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_speed_integrated, CORE_SPEED_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_speed_integrated, HALO_SPEED_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_speed_integrated, TOTAL_SPEED_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_velocity_vector_rtn_integrated, CORE_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_velocity_vector_rtn_integrated, HALO_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_velocity_vector_rtn_integrated, TOTAL_VELOCITY_VECTOR_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_heat_flux_magnitude_integrated, CORE_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_heat_flux_theta_integrated, CORE_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_heat_flux_phi_integrated, CORE_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_heat_flux_magnitude_integrated, HALO_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_heat_flux_theta_integrated, HALO_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_heat_flux_phi_integrated, HALO_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_heat_flux_magnitude_integrated, TOTAL_HEAT_FLUX_MAGNITUDE_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_heat_flux_theta_integrated, TOTAL_HEAT_FLUX_THETA_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_heat_flux_phi_integrated, TOTAL_HEAT_FLUX_PHI_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_t_parallel_integrated, CORE_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_t_perpendicular_integrated, CORE_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_t_parallel_integrated, HALO_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_t_perpendicular_integrated, HALO_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_t_parallel_integrated, TOTAL_T_PARALLEL_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_t_perpendicular_integrated, TOTAL_T_PERPENDICULAR_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_theta_rtn_integrated, CORE_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_phi_rtn_integrated, CORE_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_theta_rtn_integrated, HALO_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_phi_rtn_integrated, HALO_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_temperature_theta_rtn_integrated, TOTAL_TEMPERATURE_THETA_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_temperature_phi_rtn_integrated, TOTAL_TEMPERATURE_PHI_RTN_INTEGRATED_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_parallel_to_mag, CORE_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_perpendicular_to_mag, CORE_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_parallel_to_mag, HALO_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_perpendicular_to_mag, HALO_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_temperature_parallel_to_mag, TOTAL_TEMPERATURE_PARALLEL_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), total_temperature_perpendicular_to_mag, TOTAL_TEMPERATURE_PERPENDICULAR_TO_MAG_CDF_VAR_NAME, pycdf.const.CDF_REAL4)


if __name__ == '__main__':
    unittest.main()
