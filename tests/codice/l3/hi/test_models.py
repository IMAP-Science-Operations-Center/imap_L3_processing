import os
import unittest
from dataclasses import fields
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.hi.models import CodiceL2HiData, PriorityEventL2, CodiceL3HiDirectEvents, \
    CodiceHiL2SectoredIntensitiesData, CodiceHiL3PitchAngleDataProduct, CODICE_HI_NUM_L2_PRIORITIES
from tests.test_helpers import get_test_instrument_team_data_path


class TestModels(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_codice_hi_l2_data_read_from_instrument_team_cdf(self):
        l2_path = get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-direct-events_20241110_v002.cdf")
        l2_direct_event_data = CodiceL2HiData.read_from_cdf(l2_path)

        with CDF(str(l2_path)) as cdf:
            np.testing.assert_array_equal(l2_direct_event_data.epoch, cdf["epoch"])
            np.testing.assert_array_equal(l2_direct_event_data.epoch_delta_plus, cdf["epoch_delta_plus"])

            for priority_index in range(CODICE_HI_NUM_L2_PRIORITIES):
                actual_priority_event: PriorityEventL2 = l2_direct_event_data.priority_events[priority_index]

                # @formatter:off
                np.testing.assert_array_equal(actual_priority_event.data_quality, cdf[f"p{priority_index}_data_quality"])
                np.testing.assert_array_equal(actual_priority_event.multi_flag, cdf[f"p{priority_index}_multi_flag"])
                np.testing.assert_array_equal(actual_priority_event.number_of_events, cdf[f"p{priority_index}_num_events"])
                np.testing.assert_array_equal(actual_priority_event.ssd_energy, cdf[f"p{priority_index}_ssd_energy"])
                np.testing.assert_array_equal(actual_priority_event.ssd_id, cdf[f"p{priority_index}_ssd_id"])
                np.testing.assert_array_equal(actual_priority_event.spin_angle, cdf[f"p{priority_index}_spin_sector"])
                np.testing.assert_array_equal(actual_priority_event.spin_number, cdf[f"p{priority_index}_spin_number"])
                np.testing.assert_array_equal(actual_priority_event.time_of_flight, cdf[f"p{priority_index}_tof"])
                np.testing.assert_array_equal(actual_priority_event.type, cdf[f"p{priority_index}_type"])

                np.testing.assert_array_equal(actual_priority_event.ssd_energy_plus, cdf[f"p{priority_index}_ssd_energy_plus"])
                np.testing.assert_array_equal(actual_priority_event.ssd_energy_minus, cdf[f"p{priority_index}_ssd_energy_minus"])
                # @formatter:on

    def test_codice_l3_hi_direct_event_data_products(self):
        rng = np.random.default_rng()

        expected_epoch = np.array([datetime.now(), datetime.now() + timedelta(days=1)])
        expected_epoch_delta = np.full(expected_epoch.shape, 1)
        number_of_priority_events = 6
        event_buffer_size = 3

        (expected_data_quality,
         expected_multi_flag,
         expected_num_events,
         expected_ssd_energy,
         expected_ssd_energy_plus,
         expected_ssd_energy_minus,
         expected_ssd_id,
         expected_spin_angle,
         expected_spin_number,
         expected_tof,
         expected_type,
         expected_energy_per_nuc,
         expected_estimated_mass) = \
            [rng.random((len(expected_epoch), number_of_priority_events, event_buffer_size)) for _ in range(13)]

        expected_priority_index = np.arange(number_of_priority_events)
        expected_event_index = np.arange(event_buffer_size)
        expected_priority_index_label = np.array([str(i) for i in range(number_of_priority_events)])
        expected_event_index_label = np.array([str(i) for i in range(event_buffer_size)])

        l3_data_product = CodiceL3HiDirectEvents(Mock(), expected_epoch,
                                                 expected_epoch_delta,
                                                 expected_data_quality,
                                                 expected_multi_flag,
                                                 expected_num_events,
                                                 expected_ssd_energy,
                                                 expected_ssd_energy_plus,
                                                 expected_ssd_energy_minus,
                                                 expected_ssd_id,
                                                 expected_spin_angle,
                                                 expected_spin_number,
                                                 expected_tof,
                                                 expected_type,
                                                 expected_energy_per_nuc,
                                                 expected_estimated_mass)

        data_product_variables = l3_data_product.to_data_product_variables()
        non_parent_fields = [f for f in fields(CodiceL3HiDirectEvents) if
                             f.name in CodiceL3HiDirectEvents.__annotations__]

        self.assertEqual(len(data_product_variables), len(non_parent_fields))

        np.testing.assert_array_equal(l3_data_product.priority_index, expected_priority_index)
        np.testing.assert_array_equal(l3_data_product.event_index, expected_event_index)
        np.testing.assert_array_equal(l3_data_product.priority_index_label, expected_priority_index_label)
        np.testing.assert_array_equal(l3_data_product.event_index_label, expected_event_index_label)

        for data_product_variable in data_product_variables:
            np.testing.assert_array_equal(data_product_variable.value,
                                          getattr(l3_data_product, data_product_variable.name))

    def test_codice_hi_l3_pitch_angle_to_data_product(self):
        expected_variables = []

        epoch_data = np.array([datetime.now()])
        energy_data = np.array([100, 200])
        pitch_angle = np.array([100, 200])
        gyrophase = np.array([100, 200])
        pitch_angle_size = len(epoch_data) * len(energy_data) * len(pitch_angle)
        pitch_angle_and_gyrophase_size = len(epoch_data) * len(energy_data) * len(pitch_angle) * len(gyrophase)

        inputted_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([10]),
            "energy_h": np.array([101, 103]),
            "energy_h_delta": np.array([101, 103]) + 1,
            "energy_cno": np.array([101, 103]) + 2,
            "energy_cno_delta": np.array([101, 103]) + 3,
            "energy_fe": np.array([101, 103]) + 4,
            "energy_fe_delta": np.array([101, 103]) + 5,
            "energy_he3he4": np.array([101, 103]) + 6,
            "energy_he3he4_delta": np.array([101, 103]) + 7,
            "pitch_angle": pitch_angle,
            "pitch_angle_delta": np.array([100, 200]),
            "gyrophase": gyrophase,
            "gyrophase_delta": np.array([100, 200]),
            "h_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                              len(pitch_angle)) + 1,
            "h_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 2,
            "he4_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                                len(pitch_angle)),
            "he4_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 3,
            "o_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                              len(pitch_angle)) + 4,
            "o_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 5,
            "fe_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                               len(pitch_angle)) + 6,
            "fe_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 7,
        }

        data_product = CodiceHiL3PitchAngleDataProduct(
            input_metadata=Mock(),
            **inputted_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        for input_variable, actual_data_product_variable in zip(inputted_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_l2_sectored_intensities_read_from_instrument_team_cdf(self):
        l2_path = get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-sectored_20241110_v002.cdf")
        l2_sectored_data = CodiceHiL2SectoredIntensitiesData.read_from_cdf(l2_path)

        with CDF(str(l2_path)) as cdf:
            np.testing.assert_array_equal(l2_sectored_data.epoch, cdf["epoch"])
            expected_epoch_delta = np.array([timedelta(seconds=ns / 1e9) for ns in cdf["epoch_delta_plus"][...]])
            np.testing.assert_array_equal(l2_sectored_data.epoch_delta_plus, expected_epoch_delta)
            np.testing.assert_array_equal(l2_sectored_data.spin_sector_index, cdf['spin_sector_index'])
            np.testing.assert_array_equal(l2_sectored_data.ssd_index, cdf['ssd_index'])

            np.testing.assert_array_equal(l2_sectored_data.data_quality, cdf['data_quality'])

            np.testing.assert_array_equal(l2_sectored_data.h_intensities, cdf['h'])
            np.testing.assert_array_equal(l2_sectored_data.energy_h, cdf['energy_h'])
            np.testing.assert_array_equal(l2_sectored_data.energy_h_delta, cdf['energy_h_delta'])

            np.testing.assert_array_equal(l2_sectored_data.cno_intensities, cdf['cno'])
            np.testing.assert_array_equal(l2_sectored_data.energy_cno, cdf['energy_cno'])
            np.testing.assert_array_equal(l2_sectored_data.energy_cno_delta, cdf['energy_cno_delta'])

            np.testing.assert_array_equal(l2_sectored_data.fe_intensities, cdf['fe'])
            np.testing.assert_array_equal(l2_sectored_data.energy_fe, cdf['energy_fe'])
            np.testing.assert_array_equal(l2_sectored_data.energy_fe_delta, cdf['energy_fe_delta'])

            np.testing.assert_array_equal(l2_sectored_data.he3he4_intensities, cdf['he3he4'])
            np.testing.assert_array_equal(l2_sectored_data.energy_he3he4, cdf['energy_he3he4'])
            np.testing.assert_array_equal(l2_sectored_data.energy_he3he4_delta, cdf['energy_he3he4_delta'])

    def _create_l2_priority_event(self):
        return PriorityEventL2(data_quality=np.array([]),
                               multi_flag=np.array([]),
                               spin_angle=np.array([]),
                               number_of_events=np.array([]),
                               ssd_id=np.array([]),
                               ssd_energy=np.array([]),
                               type=np.array([]),
                               spin_number=np.array([]),
                               time_of_flight=np.array([]))
