import unittest
from datetime import timedelta
from unittest.mock import Mock

from imap_processing.swe.l3.models import SweL3Data, EPOCH_CDF_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME, ENERGY_CDF_VAR_NAME, \
    ENERGY_DELTA_PLUS_CDF_VAR_NAME, ENERGY_DELTA_MINUS_CDF_VAR_NAME, PITCH_ANGLE_CDF_VAR_NAME, \
    PITCH_ANGLE_DELTA_CDF_VAR_NAME, FLUX_BY_PITCH_ANGLE_CDF_VAR_NAME, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_data_to_product_variables(self):
        epoch = Mock()
        epoch_delta = [timedelta(days=1), timedelta(seconds=40)]
        energy = Mock()
        energy_delta_plus = Mock()
        energy_delta_minus = Mock()
        pitch_angle = Mock()
        pitch_angle_delta = Mock()
        flux_by_pitch_angle = Mock()
        psd_by_pitch_angle = Mock()

        data = SweL3Data(epoch=epoch,
                         epoch_delta=epoch_delta,
                         energy=energy,
                         energy_delta_plus=energy_delta_plus,
                         energy_delta_minus=energy_delta_minus,
                         pitch_angle=pitch_angle,
                         pitch_angle_delta=pitch_angle_delta,
                         flux_by_pitch_angle=flux_by_pitch_angle,
                         phase_space_density_by_pitch_angle=psd_by_pitch_angle,
                         input_metadata=Mock())

        variables = data.to_data_product_variables()
        self.assertEqual(9, len(variables))

        variables = iter(variables)
        # @formatter:off
        self.assert_variable_attributes(next(variables), epoch, EPOCH_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), [86400*1e9, 40*1e9], EPOCH_DELTA_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), energy, ENERGY_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), energy_delta_plus, ENERGY_DELTA_PLUS_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), energy_delta_minus, ENERGY_DELTA_MINUS_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), pitch_angle, PITCH_ANGLE_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), pitch_angle_delta, PITCH_ANGLE_DELTA_CDF_VAR_NAME, expected_record_varying=False)
        self.assert_variable_attributes(next(variables), flux_by_pitch_angle, FLUX_BY_PITCH_ANGLE_CDF_VAR_NAME)
        self.assert_variable_attributes(next(variables), psd_by_pitch_angle, PHASE_SPACE_DENSITY_BY_PITCH_ANGLE_CDF_VAR_NAME)


if __name__ == '__main__':
    unittest.main()
