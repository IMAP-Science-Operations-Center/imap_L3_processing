import os
import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.models import CodiceL1aHiData, PriorityEventL1, CodiceL2HiDataProduct, EPOCH_VAR_NAME, \
    DATA_QUALITY_VAR_NAME, NUMBER_OF_EVENTS_VAR_NAME, MULTI_FLAG_VAR_NAME, ENERGY_RANGE_VAR_NAME, ENERGY_VAR_NAME, \
    SPIN_SECTOR_VAR_NAME, SPIN_NUMBER_VAR_NAME, TIME_OF_FLIGHT_VAR_NAME, PRIORITY_VAR_NAME


class TestModels(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_codice_hit_l1_data_read_cdf(self):
        rng = np.random.default_rng()
        pathname = "test_cdf"
        with CDF(pathname, '') as cdf:
            cdf.col_major(True)

            epoch = np.array([datetime(2000, 1, 1)])
            p0_data_quality = rng.random((len(epoch)))
            p0_erge = rng.random((len(epoch), 10000))
            p0_multi_flag = rng.random((len(epoch), 10000))
            p0_num_of_events = rng.random((len(epoch)))
            p0_ssd_energy = rng.random((len(epoch), 10000))
            p0_ssd_id = rng.random((len(epoch), 10000))
            p0_spin_angle = rng.random((len(epoch), 10000))
            p0_spin_number = rng.random((len(epoch), 10000))
            p0_tof = rng.random((len(epoch), 10000))
            p0_type = rng.random((len(epoch), 10000))
            p1_data_quality = rng.random((len(epoch)))
            p1_erge = rng.random((len(epoch), 10000))
            p1_multi_flag = rng.random((len(epoch), 10000))
            p1_num_of_events = rng.random((len(epoch)))
            p1_ssd_energy = rng.random((len(epoch), 10000))
            p1_ssd_id = rng.random((len(epoch), 10000))
            p1_spin_angle = rng.random((len(epoch), 10000))
            p1_spin_number = rng.random((len(epoch), 10000))
            p1_tof = rng.random((len(epoch), 10000))
            p1_type = rng.random((len(epoch), 10000))
            p2_data_quality = rng.random((len(epoch)))
            p2_erge = rng.random((len(epoch), 10000))
            p2_multi_flag = rng.random((len(epoch), 10000))
            p2_num_of_events = rng.random((len(epoch)))
            p2_ssd_energy = rng.random((len(epoch), 10000))
            p2_ssd_id = rng.random((len(epoch), 10000))
            p2_spin_angle = rng.random((len(epoch), 10000))
            p2_spin_number = rng.random((len(epoch), 10000))
            p2_tof = rng.random((len(epoch), 10000))
            p2_type = rng.random((len(epoch), 10000))
            p3_data_quality = rng.random((len(epoch)))
            p3_erge = rng.random((len(epoch), 10000))
            p3_multi_flag = rng.random((len(epoch), 10000))
            p3_num_of_events = rng.random((len(epoch)))
            p3_ssd_energy = rng.random((len(epoch), 10000))
            p3_ssd_id = rng.random((len(epoch), 10000))
            p3_spin_angle = rng.random((len(epoch), 10000))
            p3_spin_number = rng.random((len(epoch), 10000))
            p3_tof = rng.random((len(epoch), 10000))
            p3_type = rng.random((len(epoch), 10000))
            p4_data_quality = rng.random((len(epoch)))
            p4_erge = rng.random((len(epoch), 10000))
            p4_multi_flag = rng.random((len(epoch), 10000))
            p4_num_of_events = rng.random((len(epoch)))
            p4_ssd_energy = rng.random((len(epoch), 10000))
            p4_ssd_id = rng.random((len(epoch), 10000))
            p4_spin_angle = rng.random((len(epoch), 10000))
            p4_spin_number = rng.random((len(epoch), 10000))
            p4_tof = rng.random((len(epoch), 10000))
            p4_type = rng.random((len(epoch), 10000))
            p5_data_quality = rng.random((len(epoch)))
            p5_erge = rng.random((len(epoch), 10000))
            p5_multi_flag = rng.random((len(epoch), 10000))
            p5_num_of_events = rng.random((len(epoch)))
            p5_ssd_energy = rng.random((len(epoch), 10000))
            p5_ssd_id = rng.random((len(epoch), 10000))
            p5_spin_angle = rng.random((len(epoch), 10000))
            p5_spin_number = rng.random((len(epoch), 10000))
            p5_tof = rng.random((len(epoch), 10000))
            p5_type = rng.random((len(epoch), 10000))

            cdf["P0_DataQuality"] = p0_data_quality
            cdf["P0_ERGE"] = p0_erge
            cdf["P0_MultiFlag"] = p0_multi_flag
            cdf["P0_NumEvents"] = p0_num_of_events
            cdf["P0_SSDEnergy"] = p0_ssd_energy
            cdf["P0_SSD_ID"] = p0_ssd_id
            cdf["P0_SpinAngle"] = p0_spin_angle
            cdf["P0_SpinNumber"] = p0_spin_number
            cdf["P0_TOF"] = p0_tof
            cdf["P0_Type"] = p0_type
            cdf["P1_DataQuality"] = p1_data_quality
            cdf["P1_ERGE"] = p1_erge
            cdf["P1_MultiFlag"] = p1_multi_flag
            cdf["P1_NumEvents"] = p1_num_of_events
            cdf["P1_SSDEnergy"] = p1_ssd_energy
            cdf["P1_SSD_ID"] = p1_ssd_id
            cdf["P1_SpinAngle"] = p1_spin_angle
            cdf["P1_SpinNumber"] = p1_spin_number
            cdf["P1_TOF"] = p1_tof
            cdf["P1_Type"] = p1_type
            cdf["P2_DataQuality"] = p2_data_quality
            cdf["P2_ERGE"] = p2_erge
            cdf["P2_MultiFlag"] = p2_multi_flag
            cdf["P2_NumEvents"] = p2_num_of_events
            cdf["P2_SSDEnergy"] = p2_ssd_energy
            cdf["P2_SSD_ID"] = p2_ssd_id
            cdf["P2_SpinAngle"] = p2_spin_angle
            cdf["P2_SpinNumber"] = p2_spin_number
            cdf["P2_TOF"] = p2_tof
            cdf["P2_Type"] = p2_type
            cdf["P3_DataQuality"] = p3_data_quality
            cdf["P3_ERGE"] = p3_erge
            cdf["P3_MultiFlag"] = p3_multi_flag
            cdf["P3_NumEvents"] = p3_num_of_events
            cdf["P3_SSDEnergy"] = p3_ssd_energy
            cdf["P3_SSD_ID"] = p3_ssd_id
            cdf["P3_SpinAngle"] = p3_spin_angle
            cdf["P3_SpinNumber"] = p3_spin_number
            cdf["P3_TOF"] = p3_tof
            cdf["P3_Type"] = p3_type
            cdf["P4_DataQuality"] = p4_data_quality
            cdf["P4_ERGE"] = p4_erge
            cdf["P4_MultiFlag"] = p4_multi_flag
            cdf["P4_NumEvents"] = p4_num_of_events
            cdf["P4_SSDEnergy"] = p4_ssd_energy
            cdf["P4_SSD_ID"] = p4_ssd_id
            cdf["P4_SpinAngle"] = p4_spin_angle
            cdf["P4_SpinNumber"] = p4_spin_number
            cdf["P4_TOF"] = p4_tof
            cdf["P4_Type"] = p4_type
            cdf["P5_DataQuality"] = p5_data_quality
            cdf["P5_ERGE"] = p5_erge
            cdf["P5_MultiFlag"] = p5_multi_flag
            cdf["P5_NumEvents"] = p5_num_of_events
            cdf["P5_SSDEnergy"] = p5_ssd_energy
            cdf["P5_SSD_ID"] = p5_ssd_id
            cdf["P5_SpinAngle"] = p5_spin_angle
            cdf["P5_SpinNumber"] = p5_spin_number
            cdf["P5_TOF"] = p5_tof
            cdf["P5_Type"] = p5_type

            cdf["epoch"] = epoch

        for path in [pathname]:
            with self.subTest(path=path):
                result: CodiceL1aHiData = CodiceL1aHiData.read_from_cdf(path)

                priority_event_0: PriorityEventL1 = result.priority_event_0
                np.testing.assert_array_equal(priority_event_0.number_of_events, p0_num_of_events)
                np.testing.assert_array_equal(priority_event_0.energy_range, p0_erge)
                np.testing.assert_array_equal(priority_event_0.ssd_id, p0_ssd_id)
                np.testing.assert_array_equal(priority_event_0.ssd_energy, p0_ssd_energy)
                np.testing.assert_array_equal(priority_event_0.type, p0_type)
                np.testing.assert_array_equal(priority_event_0.data_quality, p0_data_quality)
                np.testing.assert_array_equal(priority_event_0.multi_flag, p0_multi_flag)
                np.testing.assert_array_equal(priority_event_0.spin_angle, p0_spin_angle)
                np.testing.assert_array_equal(priority_event_0.spin_number, p0_spin_number)
                np.testing.assert_array_equal(priority_event_0.time_of_flight, p0_tof)

                priority_event_1: PriorityEventL1 = result.priority_event_1
                np.testing.assert_array_equal(priority_event_1.number_of_events, p1_num_of_events)
                np.testing.assert_array_equal(priority_event_1.energy_range, p1_erge)
                np.testing.assert_array_equal(priority_event_1.ssd_id, p1_ssd_id)
                np.testing.assert_array_equal(priority_event_1.ssd_energy, p1_ssd_energy)
                np.testing.assert_array_equal(priority_event_1.type, p1_type)
                np.testing.assert_array_equal(priority_event_1.data_quality, p1_data_quality)
                np.testing.assert_array_equal(priority_event_1.multi_flag, p1_multi_flag)
                np.testing.assert_array_equal(priority_event_1.spin_angle, p1_spin_angle)
                np.testing.assert_array_equal(priority_event_1.spin_number, p1_spin_number)
                np.testing.assert_array_equal(priority_event_1.time_of_flight, p1_tof)

                priority_event_2: PriorityEventL1 = result.priority_event_2
                np.testing.assert_array_equal(priority_event_2.number_of_events, p2_num_of_events)
                np.testing.assert_array_equal(priority_event_2.energy_range, p2_erge)
                np.testing.assert_array_equal(priority_event_2.ssd_id, p2_ssd_id)
                np.testing.assert_array_equal(priority_event_2.ssd_energy, p2_ssd_energy)
                np.testing.assert_array_equal(priority_event_2.type, p2_type)
                np.testing.assert_array_equal(priority_event_2.data_quality, p2_data_quality)
                np.testing.assert_array_equal(priority_event_2.multi_flag, p2_multi_flag)
                np.testing.assert_array_equal(priority_event_2.spin_angle, p2_spin_angle)
                np.testing.assert_array_equal(priority_event_2.spin_number, p2_spin_number)
                np.testing.assert_array_equal(priority_event_2.time_of_flight, p2_tof)

                priority_event_3: PriorityEventL1 = result.priority_event_3
                np.testing.assert_array_equal(priority_event_3.number_of_events, p3_num_of_events)
                np.testing.assert_array_equal(priority_event_3.energy_range, p3_erge)
                np.testing.assert_array_equal(priority_event_3.ssd_id, p3_ssd_id)
                np.testing.assert_array_equal(priority_event_3.ssd_energy, p3_ssd_energy)
                np.testing.assert_array_equal(priority_event_3.type, p3_type)
                np.testing.assert_array_equal(priority_event_3.data_quality, p3_data_quality)
                np.testing.assert_array_equal(priority_event_3.multi_flag, p3_multi_flag)
                np.testing.assert_array_equal(priority_event_3.spin_angle, p3_spin_angle)
                np.testing.assert_array_equal(priority_event_3.spin_number, p3_spin_number)
                np.testing.assert_array_equal(priority_event_3.time_of_flight, p3_tof)

                priority_event_4: PriorityEventL1 = result.priority_event_4
                np.testing.assert_array_equal(priority_event_4.number_of_events, p4_num_of_events)
                np.testing.assert_array_equal(priority_event_4.energy_range, p4_erge)
                np.testing.assert_array_equal(priority_event_4.ssd_id, p4_ssd_id)
                np.testing.assert_array_equal(priority_event_4.ssd_energy, p4_ssd_energy)
                np.testing.assert_array_equal(priority_event_4.type, p4_type)
                np.testing.assert_array_equal(priority_event_4.data_quality, p4_data_quality)
                np.testing.assert_array_equal(priority_event_4.multi_flag, p4_multi_flag)
                np.testing.assert_array_equal(priority_event_4.spin_angle, p4_spin_angle)
                np.testing.assert_array_equal(priority_event_4.spin_number, p4_spin_number)
                np.testing.assert_array_equal(priority_event_4.time_of_flight, p4_tof)

                priority_event_5: PriorityEventL1 = result.priority_event_5
                np.testing.assert_array_equal(priority_event_5.number_of_events, p5_num_of_events)
                np.testing.assert_array_equal(priority_event_5.energy_range, p5_erge)
                np.testing.assert_array_equal(priority_event_5.ssd_id, p5_ssd_id)
                np.testing.assert_array_equal(priority_event_5.ssd_energy, p5_ssd_energy)
                np.testing.assert_array_equal(priority_event_5.type, p5_type)
                np.testing.assert_array_equal(priority_event_5.data_quality, p5_data_quality)
                np.testing.assert_array_equal(priority_event_5.multi_flag, p5_multi_flag)
                np.testing.assert_array_equal(priority_event_5.spin_angle, p5_spin_angle)
                np.testing.assert_array_equal(priority_event_5.spin_number, p5_spin_number)
                np.testing.assert_array_equal(priority_event_5.time_of_flight, p5_tof)

    def test_to_data_product_variables(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2000, 1, 1)])
        priority = np.array([0, 1, 2, 3, 4, 5])
        data_quality = rng.random((len(epoch), len(priority)))
        energy_range = rng.random((len(epoch), len(priority), 10000))
        multi_flag = rng.random((len(epoch), len(priority), 10000))
        number_of_events = rng.random((len(epoch), len(priority)))
        energy = rng.random((len(epoch), len(priority), 10000))
        spin_sector = rng.random((len(epoch), len(priority), 10000))
        spin_number = rng.random((len(epoch), len(priority), 10000))
        time_of_flight = rng.random((len(epoch), len(priority), 10000))

        data_product = CodiceL2HiDataProduct(
            epoch=epoch,
            data_quality=data_quality,
            energy_range=energy_range,
            multi_flag=multi_flag,
            number_of_events=number_of_events,
            energy=energy,
            spin_sector=spin_sector,
            spin_number=spin_number,
            time_of_flight=time_of_flight,
            priority=priority,
            input_metadata=Mock(),
        )

        data_product_variables = data_product.to_data_product_variables()

        cases = [
            (EPOCH_VAR_NAME, epoch),
            (DATA_QUALITY_VAR_NAME, data_quality),
            (ENERGY_RANGE_VAR_NAME, energy_range),
            (MULTI_FLAG_VAR_NAME, multi_flag),
            (NUMBER_OF_EVENTS_VAR_NAME, number_of_events),
            (ENERGY_VAR_NAME, energy),
            (SPIN_SECTOR_VAR_NAME, spin_sector),
            (SPIN_NUMBER_VAR_NAME, spin_number),
            (TIME_OF_FLIGHT_VAR_NAME, time_of_flight),
            (PRIORITY_VAR_NAME, priority),
        ]

        for index, (var_name, value) in enumerate(cases):
            self.assertEqual(data_product_variables[index].name, var_name)
            np.testing.assert_array_equal(data_product_variables[index].value, value)
