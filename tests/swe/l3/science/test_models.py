import unittest
from unittest.mock import Mock

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
    INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME
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
        intensity_by_pitch_angle = Mock()
        intensity_by_pitch_angle_and_gyrophase = Mock()
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
                         intensity_by_pitch_angle=intensity_by_pitch_angle,
                         intensity_by_pitch_angle_and_gyrophase=intensity_by_pitch_angle_and_gyrophase,
                         spacecraft_potential=spacecraft_potential,
                         core_halo_breakpoint=core_halo_breakpoint,
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
                         input_metadata=Mock())

        variables = data.to_data_product_variables()
        self.assertEqual(32, len(variables))

        variables = iter(variables)
        # @formatter:off
        self.assert_variable_attributes(
            next(variables), epoch, EPOCH_CDF_VAR_NAME, pycdf.const.CDF_TIME_TT2000)
        self.assert_variable_attributes(
            next(variables), [30_000_000_000, 30_000_000_000], EPOCH_DELTA_CDF_VAR_NAME, pycdf.const.CDF_INT8)
        self.assert_variable_attributes(
            next(variables), energy, ENERGY_CDF_VAR_NAME, pycdf.const.CDF_REAL4, expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), energy_delta_plus, ENERGY_DELTA_PLUS_CDF_VAR_NAME, pycdf.const.CDF_REAL4,
            expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), energy_delta_minus, ENERGY_DELTA_MINUS_CDF_VAR_NAME, pycdf.const.CDF_REAL4,
            expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), pitch_angle, PITCH_ANGLE_CDF_VAR_NAME, pycdf.const.CDF_REAL4,
            expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), pitch_angle_delta, PITCH_ANGLE_DELTA_CDF_VAR_NAME, pycdf.const.CDF_REAL4,
            expected_record_varying=False)
        self.assert_variable_attributes(
            next(variables), psd_by_pitch_angle, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum, ENERGY_SPECTRUM_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum_inbound, ENERGY_SPECTRUM_INBOUND_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), energy_spectrum_outbound, ENERGY_SPECTRUM_OUTBOUND_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), intensity_by_pitch_angle, INTENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), intensity_by_pitch_angle_and_gyrophase,
            INTENSITY_BY_PITCH_ANGLE_AND_GYROPHASE_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
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
            next(variables), core_temperature_phi_rtn_fit, CORE_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_phi_rtn_fit, HALO_TEMPERATURE_PHI_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_temperature_theta_rtn_fit, CORE_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_temperature_theta_rtn_fit, HALO_TEMPERATURE_THETA_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_speed_fit, CORE_SPEED_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_speed_fit, HALO_SPEED_FIT_CDF_VAR_NAME, pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), core_velocity_vector_rtn_fit, CORE_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)
        self.assert_variable_attributes(
            next(variables), halo_velocity_vector_rtn_fit, HALO_VELOCITY_VECTOR_RTN_FIT_CDF_VAR_NAME,
            pycdf.const.CDF_REAL4)


if __name__ == '__main__':
    unittest.main()
