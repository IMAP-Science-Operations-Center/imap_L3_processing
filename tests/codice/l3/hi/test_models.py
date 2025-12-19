import os
import unittest
from dataclasses import fields
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.hi.models import CodiceL2HiDirectEventData, CodiceL3HiDirectEvents, \
    CodiceHiL2SectoredIntensitiesData, CodiceHiL3PitchAngleDataProduct
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path


class TestModels(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_codice_hi_l2_data_read_from_instrument_team_cdf(self):
        l2_path = get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-direct-events_20250814_v001.cdf")
        l2_direct_event_data = CodiceL2HiDirectEventData.read_from_cdf(l2_path)

        with CDF(str(l2_path)) as cdf:
            # @formatter:off
            np.testing.assert_array_equal(l2_direct_event_data.epoch, cdf["epoch"])
            np.testing.assert_array_equal(l2_direct_event_data.epoch_delta_plus, cdf["epoch_delta_plus"])
            np.testing.assert_array_equal(l2_direct_event_data.ssd_energy, cdf[f"ssd_energy"])
            expected_ssd_id = cdf["ssd_id"][...]
            np.testing.assert_array_equal(l2_direct_event_data.ssd_id, expected_ssd_id)

            expected_energy_per_nuc = cdf["energy_per_nuc"][...]
            expected_energy_per_nuc[expected_energy_per_nuc==-1e31] = np.nan
            np.testing.assert_array_equal(l2_direct_event_data.energy_per_nuc, expected_energy_per_nuc)
            
            expected_spin_angle = cdf["spin_angle"][...]
            expected_spin_angle[expected_spin_angle==-1e31] = np.nan
            np.testing.assert_array_equal(l2_direct_event_data.spin_angle, expected_spin_angle)

            expected_spin_number = cdf["spin_number"][...]
            np.testing.assert_array_equal(l2_direct_event_data.spin_number, expected_spin_number)

            expected_time_of_flight = cdf["tof"][...]
            expected_time_of_flight[expected_time_of_flight==-1e31] = np.nan
            np.testing.assert_array_equal(l2_direct_event_data.time_of_flight, expected_time_of_flight)

            np.testing.assert_array_equal(l2_direct_event_data.number_of_events, cdf[f"num_events"])
            np.testing.assert_array_equal(l2_direct_event_data.data_quality, cdf[f"data_quality"])
            np.testing.assert_array_equal(l2_direct_event_data.multi_flag, cdf[f"multi_flag"])

            np.testing.assert_array_equal(l2_direct_event_data.type, cdf[f"type"])
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
         expected_ssd_id,
         expected_spin_angle,
         expected_spin_number,
         expected_tof,
         expected_type,
         expected_energy_per_nuc,
         expected_estimated_mass) = \
            [rng.random((len(expected_epoch), number_of_priority_events, event_buffer_size)) for _ in range(11)]

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
        epoch_data = np.array([datetime.now()])
        energy_data = np.array([100, 200])
        pitch_angle = np.array([100, 200])
        gyrophase = np.array([100, 200])
        pitch_angle_size = len(epoch_data) * len(energy_data) * len(pitch_angle)
        pitch_angle_and_gyrophase_size = len(epoch_data) * len(energy_data) * len(pitch_angle) * len(gyrophase)

        energy_h_data = np.array([101, 103])
        energy_cno_data = np.array([101, 103]) + 2
        energy_fe_data = np.array([101, 103]) + 4
        energy_he3he4_data = np.array([101, 103]) + 6
        inputted_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([timedelta(seconds=10)]),
            "energy_h": energy_h_data,
            "energy_h_plus": np.array([101, 103]) + 1,
            "energy_h_minus": np.array([101, 103]) + 11,
            "energy_cno": energy_cno_data,
            "energy_cno_plus": np.array([101, 103]) + 3,
            "energy_cno_minus": np.array([101, 103]) + 33,
            "energy_fe": energy_fe_data,
            "energy_fe_plus": np.array([101, 103]) + 5,
            "energy_fe_minus": np.array([101, 103]) + 55,
            "energy_he3he4": energy_he3he4_data,
            "energy_he3he4_plus": np.array([101, 103]) + 7,
            "energy_he3he4_minus": np.array([101, 103]) + 77,
            "pitch_angle": pitch_angle,
            "pitch_angle_delta": np.array([100, 200]),
            "gyrophase": gyrophase,
            "gyrophase_delta": np.array([100, 200]),
            "h_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                              len(pitch_angle)) + 1,
            "h_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 2,
            "he3he4_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                                len(pitch_angle)),
            "he3he4_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
                len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 3,
            "cno_intensity_by_pitch_angle": np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data),
                                                                                len(pitch_angle)) + 4,
            "cno_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(
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

        np.testing.assert_array_equal(data_product.energy_h_label, np.array([str(v) for v in energy_h_data]))
        np.testing.assert_array_equal(data_product.energy_cno_label, np.array([str(v) for v in energy_cno_data]))
        np.testing.assert_array_equal(data_product.energy_fe_label, np.array([str(v) for v in energy_fe_data]))
        np.testing.assert_array_equal(data_product.energy_he3he4_label, np.array([str(v) for v in energy_he3he4_data]))
        np.testing.assert_array_equal(data_product.pitch_angle_label, np.array([str(v) for v in pitch_angle]))
        np.testing.assert_array_equal(data_product.gyrophase_label, np.array([str(v) for v in gyrophase]))

        actual_data_product_variables = data_product.to_data_product_variables()

        non_parent_fields = [f for f in fields(CodiceHiL3PitchAngleDataProduct)
                             if f.name in CodiceHiL3PitchAngleDataProduct.__annotations__]
        data_product_fields = [f.name for f in non_parent_fields]

        self.assertEqual(len(actual_data_product_variables), len(non_parent_fields))
        for data_product_var in actual_data_product_variables:
            self.assertIn(data_product_var.name, data_product_fields)
            if data_product_var.name != 'epoch_delta':
                np.testing.assert_array_equal(data_product_var.value, getattr(data_product, data_product_var.name))
            else:
                np.testing.assert_array_equal(data_product_var.value, np.array([10 * 1e9]))

    def test_l2_sectored_intensities_read_from_sdc_cdf(self):
        l2_path = get_test_instrument_team_data_path(
            "codice/hi/imap_codice_l2_hi-sectored_20250814_v002.cdf")
        l2_sectored_data = CodiceHiL2SectoredIntensitiesData.read_from_cdf(l2_path)

        with CDF(str(l2_path)) as cdf:
            np.testing.assert_array_equal(l2_sectored_data.epoch, cdf["epoch"])
            expected_epoch_delta = np.array([timedelta(seconds=ns / 1e9) for ns in cdf["epoch_delta_plus"][...]])
            np.testing.assert_array_equal(l2_sectored_data.epoch_delta_plus, expected_epoch_delta)
            np.testing.assert_array_equal(l2_sectored_data.spin_angles, cdf['spin_angles'][...])
            np.testing.assert_array_equal(l2_sectored_data.elevation_angle, cdf['elevation_angle'])

            np.testing.assert_array_equal(l2_sectored_data.data_quality, cdf['data_quality'])

            np.testing.assert_array_equal(l2_sectored_data.h_intensities, cdf['h'])
            np.testing.assert_array_equal(l2_sectored_data.energy_h, cdf['energy_h'])
            np.testing.assert_array_equal(l2_sectored_data.energy_h_plus, cdf['energy_h_plus'])
            np.testing.assert_array_equal(l2_sectored_data.energy_h_minus, cdf['energy_h_minus'])

            np.testing.assert_array_equal(l2_sectored_data.cno_intensities, cdf['cno'])
            np.testing.assert_array_equal(l2_sectored_data.energy_cno, cdf['energy_cno'])
            np.testing.assert_array_equal(l2_sectored_data.energy_cno_plus, cdf['energy_cno_plus'])
            np.testing.assert_array_equal(l2_sectored_data.energy_cno_minus, cdf['energy_cno_minus'])

            np.testing.assert_array_equal(l2_sectored_data.fe_intensities, cdf['fe'])
            np.testing.assert_array_equal(l2_sectored_data.energy_fe, cdf['energy_fe'])
            np.testing.assert_array_equal(l2_sectored_data.energy_fe_plus, cdf['energy_fe_plus'])
            np.testing.assert_array_equal(l2_sectored_data.energy_fe_minus, cdf['energy_fe_minus'])

            np.testing.assert_array_equal(l2_sectored_data.he3he4_intensities, cdf['he3he4'])
            np.testing.assert_array_equal(l2_sectored_data.energy_he3he4, cdf['energy_he3he4'])
            np.testing.assert_array_equal(l2_sectored_data.energy_he3he4_plus, cdf['energy_he3he4_plus'])
            np.testing.assert_array_equal(l2_sectored_data.energy_he3he4_minus, cdf['energy_he3he4_minus'])

    def test_l2_sectored_intensities_read_from_instrument_team_cdf_handles_fill_values(self):
        all_fill_l2_sectored_intensities_path = \
            get_test_data_path('codice/imap_codice_l2_hi-sectored_20250814_v002-all-fill.cdf')
        l2_sectored_intensities = CodiceHiL2SectoredIntensitiesData.read_from_cdf(all_fill_l2_sectored_intensities_path)
        with CDF(str(all_fill_l2_sectored_intensities_path)) as cdf:
            np.testing.assert_array_equal(l2_sectored_intensities.cno_intensities, np.full_like(cdf['cno'], np.nan))
            np.testing.assert_array_equal(l2_sectored_intensities.fe_intensities, np.full_like(cdf['fe'], np.nan))
            np.testing.assert_array_equal(l2_sectored_intensities.h_intensities, np.full_like(cdf['h'], np.nan))
            np.testing.assert_array_equal(l2_sectored_intensities.he3he4_intensities,
                                          np.full_like(cdf['he3he4'], np.nan))
