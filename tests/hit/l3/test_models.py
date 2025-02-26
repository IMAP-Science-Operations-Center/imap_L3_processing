from datetime import datetime

import numpy as np

from imap_processing.hit.l3 import models
from imap_processing.hit.l3.models import HitDirectEventDataProduct
from imap_processing.hit.l3.pha.pha_event_reader import RawPHAEvent, PHAWord, Detector, PHAExtendedHeader, StimBlock, \
    ExtendedStimHeader
from imap_processing.hit.l3.pha.science.calculate_pha import EventOutput
from imap_processing.models import DataProductVariable, UpstreamDataDependency
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def test_pha_to_data_product_variables_with_multiple_events(self):
        pha_word = PHAWord(adc_overflow=False, adc_value=11,
                           detector=Detector(layer=1, side="A", segment="1A", address=200), is_last_pha=True,
                           is_low_gain=True)

        raw_pha_event = RawPHAEvent(particle_id=1, priority_buffer_num=2, stim_tag=False, haz_tag=False, time_tag=20,
                                    a_b_side_flag=False, has_unread_adcs=True, long_event_flag=False, culling_flag=True,
                                    spare=True, pha_words=[pha_word])

        pha_word_2 = PHAWord(adc_overflow=True, adc_value=12,
                             detector=Detector(layer=2, side="B", segment="1B", address=201), is_last_pha=False,
                             is_low_gain=False)

        raw_pha_event_2 = RawPHAEvent(particle_id=2, priority_buffer_num=3, stim_tag=True, haz_tag=True, time_tag=30,
                                      a_b_side_flag=True, has_unread_adcs=False, long_event_flag=True,
                                      culling_flag=False,
                                      spare=False, pha_words=[pha_word_2],
                                      extended_header=PHAExtendedHeader(detector_flags=2, delta_e_index=2,
                                                                        e_prime_index=True),
                                      stim_block=StimBlock(stim_step=1, stim_gain=2, unused=3, a_l_stim=True),
                                      extended_stim_header=ExtendedStimHeader(dac_value=123, tbd=666))

        event_output = EventOutput(original_event=raw_pha_event, charge=10.0, energies=[1, 2, 3], total_energy=100)
        event_output_2 = EventOutput(original_event=raw_pha_event_2, charge=12.0, energies=[9, 8, 7], total_energy=200)

        input_metadata = UpstreamDataDependency(
            instrument="HIT",
            data_level="L3A",
            start_date=datetime.min,
            end_date=datetime.max,
            version="1",
            descriptor="PHA"
        )

        pha_data = HitDirectEventDataProduct(input_metadata=input_metadata,
                                             event_outputs=[event_output, event_output_2])

        expected_variables = [
            DataProductVariable(models.CHARGE_VAR_NAME, np.array([10.0, 12.0])),
            DataProductVariable(models.ENERGY_VAR_NAME, np.array([100, 200])),
            DataProductVariable(models.PARTICLE_ID_VAR_NAME, np.array([1, 2])),
            DataProductVariable(models.PRIORITY_BUFFER_ID_VAR_NAME, np.array([2, 3])),
            DataProductVariable(models.LATENCY_VAR_NAME, np.array([20, 30])),
            DataProductVariable(models.STIM_FLAG_VAR_NAME, np.array([False, True])),
            DataProductVariable(models.LONG_EVENT_FLAG_VAR_NAME, np.array([False, True])),
            DataProductVariable(models.HAZ_FLAG_VAR_NAME, np.array([False, True])),
            DataProductVariable(models.A_B_SIDE_VAR_NAME, np.array([False, True])),
            DataProductVariable(models.HAS_UNREAD_FLAG_VAR_NAME, np.array([True, False])),
            DataProductVariable(models.CULLING_FLAG_VAR_NAME, np.array([True, False])),
            DataProductVariable(models.PHA_VALUE_VAR_NAME, np.array([[11], [12]])),
            DataProductVariable(models.ENERGY_AT_DETECTOR_VAR_NAME, np.array([[1, 2, 3], [9, 8, 7]])),
            DataProductVariable(models.DETECTOR_ADDRESS_VAR_NAME, np.array([[200], [201]])),
            DataProductVariable(models.GAIN_FLAG_VAR_NAME, np.array([[True], [False]])),
            DataProductVariable(models.LAST_PHA_VAR_NAME, np.array([[True], [False]])),
            DataProductVariable(models.DETECTOR_FLAG_VAR_NAME, np.array([None, 2])),
            DataProductVariable(models.DEINDEX_VAR_NAME, np.array([None, 2])),
            DataProductVariable(models.EPINDEX_VAR_NAME, np.array([None, True])),
            DataProductVariable(models.STIM_GAIN_VAR_NAME, np.array([None, 2])),
            DataProductVariable(models.A_L_STIM_VAR_NAME, np.array([None, True])),
            DataProductVariable(models.STIM_STEP_VAR_NAME, np.array([None, 1])),
            DataProductVariable(models.DAC_VALUE_VAR_NAME, np.array([None, 123]))
        ]

        actual_variables = pha_data.to_data_product_variables()

        self.assertEqual(23, len(actual_variables))
        self.assert_variable_attributes(actual_variables[0], expected_variables[0].value, expected_variables[0].name,
                                        expected_variables[0].cdf_data_type, expected_variables[0].record_varying)
        self.assert_variable_attributes(actual_variables[1], expected_variables[1].value, expected_variables[1].name,
                                        expected_variables[1].cdf_data_type, expected_variables[1].record_varying)
        self.assert_variable_attributes(actual_variables[2], expected_variables[2].value, expected_variables[2].name,
                                        expected_variables[2].cdf_data_type, expected_variables[2].record_varying)
        self.assert_variable_attributes(actual_variables[3], expected_variables[3].value, expected_variables[3].name,
                                        expected_variables[3].cdf_data_type, expected_variables[3].record_varying)
        self.assert_variable_attributes(actual_variables[4], expected_variables[4].value, expected_variables[4].name,
                                        expected_variables[4].cdf_data_type, expected_variables[4].record_varying)
        self.assert_variable_attributes(actual_variables[5], expected_variables[5].value, expected_variables[5].name,
                                        expected_variables[5].cdf_data_type, expected_variables[5].record_varying)
        self.assert_variable_attributes(actual_variables[6], expected_variables[6].value, expected_variables[6].name,
                                        expected_variables[6].cdf_data_type, expected_variables[6].record_varying)
        self.assert_variable_attributes(actual_variables[7], expected_variables[7].value, expected_variables[7].name,
                                        expected_variables[7].cdf_data_type, expected_variables[7].record_varying)
        self.assert_variable_attributes(actual_variables[8], expected_variables[8].value, expected_variables[8].name,
                                        expected_variables[8].cdf_data_type, expected_variables[8].record_varying)
        self.assert_variable_attributes(actual_variables[9], expected_variables[9].value, expected_variables[9].name,
                                        expected_variables[9].cdf_data_type, expected_variables[9].record_varying)
        self.assert_variable_attributes(actual_variables[10], expected_variables[10].value, expected_variables[10].name,
                                        expected_variables[10].cdf_data_type, expected_variables[10].record_varying)
        self.assert_variable_attributes(actual_variables[11], expected_variables[11].value, expected_variables[11].name,
                                        expected_variables[11].cdf_data_type, expected_variables[11].record_varying)
        self.assert_variable_attributes(actual_variables[12], expected_variables[12].value, expected_variables[12].name,
                                        expected_variables[12].cdf_data_type, expected_variables[12].record_varying)
        self.assert_variable_attributes(actual_variables[13], expected_variables[13].value, expected_variables[13].name,
                                        expected_variables[13].cdf_data_type, expected_variables[13].record_varying)
        self.assert_variable_attributes(actual_variables[14], expected_variables[14].value, expected_variables[14].name,
                                        expected_variables[14].cdf_data_type, expected_variables[14].record_varying)
        self.assert_variable_attributes(actual_variables[15], expected_variables[15].value, expected_variables[15].name,
                                        expected_variables[15].cdf_data_type, expected_variables[15].record_varying)
        self.assert_variable_attributes(actual_variables[16], expected_variables[16].value, expected_variables[16].name,
                                        expected_variables[16].cdf_data_type, expected_variables[16].record_varying)

        self.assert_variable_attributes(actual_variables[17], expected_variables[17].value, expected_variables[17].name,
                                        expected_variables[17].cdf_data_type, expected_variables[17].record_varying)
        self.assert_variable_attributes(actual_variables[18], expected_variables[18].value, expected_variables[18].name,
                                        expected_variables[18].cdf_data_type, expected_variables[18].record_varying)
        self.assert_variable_attributes(actual_variables[19], expected_variables[19].value, expected_variables[19].name,
                                        expected_variables[19].cdf_data_type, expected_variables[19].record_varying)
        self.assert_variable_attributes(actual_variables[20], expected_variables[20].value, expected_variables[20].name,
                                        expected_variables[20].cdf_data_type, expected_variables[20].record_varying)
        self.assert_variable_attributes(actual_variables[21], expected_variables[21].value, expected_variables[21].name,
                                        expected_variables[21].cdf_data_type, expected_variables[21].record_varying)
        self.assert_variable_attributes(actual_variables[22], expected_variables[22].value, expected_variables[22].name,
                                        expected_variables[22].cdf_data_type, expected_variables[22].record_varying)
