import os
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.hit.l3 import models
from imap_processing.hit.l3.models import HitDirectEventDataProduct, HitL1Data
from imap_processing.models import DataProductVariable, UpstreamDataDependency
from tests.swapi.cdf_model_test_case import CdfModelTestCase


class TestModels(CdfModelTestCase):
    def setUp(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self) -> None:
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_pha_to_data_product_variables_with_multiple_events(self):
        input_metadata = UpstreamDataDependency(
            instrument="HIT",
            data_level="L3A",
            start_date=datetime.min,
            end_date=datetime.max,
            version="1",
            descriptor="direct-event"
        )

        charge = np.array([10.0, 12.0])
        energy = np.array([100, 200])
        particle = np.array([1, 2])
        priority_buffer_number = np.array([2, 3])
        latency = np.array([20, 30])
        stim_tag = np.array([False, True])
        long_event_flag = np.array([False, True])
        haz_tag = np.array([False, True])
        a_b_side = np.array([False, True])
        has_unread_adcs = np.array([True, False])
        culling_flag = np.array([True, False])
        pha_value = np.array([[11], [12]])
        energy_at_detector = np.array([[1, 2, 3], [9, 8, 7]])
        detector_address = np.array([[200], [201]])
        is_low_gain = np.array([[True], [False]])
        detector_flags = np.array([None, 2])
        deindex = np.array([None, 2])
        epindex = np.array([None, True])
        stim_gain = np.array([None, 2])
        a_l_stim = np.array([None, True])
        stim_step = np.array([None, 1])
        dac_value = np.array([None, 123])

        direct_event = HitDirectEventDataProduct(input_metadata=input_metadata,
                                                 charge=charge,
                                                 energy=energy,
                                                 particle_id=particle,
                                                 priority_buffer_number=priority_buffer_number,
                                                 latency=latency,
                                                 stim_tag=stim_tag,
                                                 long_event_flag=long_event_flag,
                                                 haz_tag=haz_tag,
                                                 a_b_side=a_b_side,
                                                 has_unread_adcs=has_unread_adcs,
                                                 culling_flag=culling_flag,
                                                 pha_value=pha_value,
                                                 energy_at_detector=energy_at_detector,
                                                 detector_address=detector_address,
                                                 is_low_gain=is_low_gain,
                                                 detector_flags=detector_flags,
                                                 deindex=deindex,
                                                 epindex=epindex,
                                                 stim_gain=stim_gain,
                                                 a_l_stim=a_l_stim,
                                                 stim_step=stim_step,
                                                 dac_value=dac_value, )

        expected_variables = [
            DataProductVariable(models.CHARGE_VAR_NAME, charge),
            DataProductVariable(models.ENERGY_VAR_NAME, energy),
            DataProductVariable(models.PARTICLE_ID_VAR_NAME, particle),
            DataProductVariable(models.PRIORITY_BUFFER_NUMBER_VAR_NAME, priority_buffer_number),
            DataProductVariable(models.LATENCY_VAR_NAME, latency),
            DataProductVariable(models.STIM_TAG_VAR_NAME, stim_tag),
            DataProductVariable(models.LONG_EVENT_FLAG_VAR_NAME, long_event_flag),
            DataProductVariable(models.HAZ_TAG_VAR_NAME, haz_tag),
            DataProductVariable(models.A_B_SIDE_VAR_NAME, a_b_side),
            DataProductVariable(models.HAS_UNREAD_FLAG_VAR_NAME, has_unread_adcs),
            DataProductVariable(models.CULLING_FLAG_VAR_NAME, culling_flag),
            DataProductVariable(models.PHA_VALUE_VAR_NAME, pha_value),
            DataProductVariable(models.ENERGY_AT_DETECTOR_VAR_NAME, energy_at_detector),
            DataProductVariable(models.DETECTOR_ADDRESS_VAR_NAME, detector_address),
            DataProductVariable(models.IS_LOW_GAIN_VAR_NAME, is_low_gain),
            DataProductVariable(models.DETECTOR_FLAGS_VAR_NAME, detector_flags),
            DataProductVariable(models.DEINDEX_VAR_NAME, deindex),
            DataProductVariable(models.EPINDEX_VAR_NAME, epindex),
            DataProductVariable(models.STIM_GAIN_VAR_NAME, stim_gain),
            DataProductVariable(models.A_L_STIM_VAR_NAME, a_l_stim),
            DataProductVariable(models.STIM_STEP_VAR_NAME, stim_step),
            DataProductVariable(models.DAC_VALUE_VAR_NAME, dac_value)
        ]

        actual_variables = direct_event.to_data_product_variables()

        self.assertEqual(22, len(actual_variables))
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

    def test_hit_l1_data_read_from_cdf(self):
        expected_epochs = np.array([datetime(year=2020, month=1, day=1), datetime(year=2020, month=1, day=1, hour=2)])
        expected_raw_binaries = ["10101011", "10101001010101"]

        pathname = 'test_cdf'
        with CDF(pathname, '') as cdf:
            cdf["epoch"] = expected_epochs
            cdf["pha_raw"] = expected_raw_binaries

        for file_path in [pathname, Path(pathname)]:
            with self.subTest(f"Can take {type(file_path)} as file path type"):
                hit_l1_data = HitL1Data.read_from_cdf(file_path)

                np.testing.assert_array_equal(hit_l1_data.epoch, expected_epochs)
                np.testing.assert_array_equal(expected_raw_binaries, hit_l1_data.event_binary)
