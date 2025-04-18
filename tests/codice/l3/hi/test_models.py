import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.hi.models import CodiceL2HiData, PriorityEventL2, CodiceL3HiDirectEvents, \
    CodiceL3HiDirectEventsBuilder, CodiceHiL2SectoredIntensitiesData, CodiceHiL3PitchAngleDataProduct


class TestModels(unittest.TestCase):
    def setUp(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def tearDown(self):
        if os.path.exists('test_cdf.cdf'):
            os.remove('test_cdf.cdf')

    def test_get_priority_events(self):
        codice_l2_data = CodiceL2HiData(epochs=np.array([]), priority_event_0=Mock(), priority_event_1=Mock(),
                                        priority_event_2=Mock(),
                                        priority_event_3=Mock(), priority_event_4=Mock(), priority_event_5=Mock())
        (actual_priority_event_0, actual_priority_event_1, actual_priority_event_2,
         actual_priority_event_3, actual_priority_event_4, actual_priority_event_5) = codice_l2_data.priority_events

        self.assertEqual(codice_l2_data.priority_event_0, actual_priority_event_0)
        self.assertEqual(codice_l2_data.priority_event_1, actual_priority_event_1)
        self.assertEqual(codice_l2_data.priority_event_2, actual_priority_event_2)
        self.assertEqual(codice_l2_data.priority_event_3, actual_priority_event_3)
        self.assertEqual(codice_l2_data.priority_event_4, actual_priority_event_4)
        self.assertEqual(codice_l2_data.priority_event_5, actual_priority_event_5)

    def test_codice_hit_l2_data_read_cdf(self):
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
                result: CodiceL2HiData = CodiceL2HiData.read_from_cdf(path)

                priority_event_0: PriorityEventL2 = result.priority_event_0
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

                priority_event_1: PriorityEventL2 = result.priority_event_1
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

                priority_event_2: PriorityEventL2 = result.priority_event_2
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

                priority_event_3: PriorityEventL2 = result.priority_event_3
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

                priority_event_4: PriorityEventL2 = result.priority_event_4
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

                priority_event_5: PriorityEventL2 = result.priority_event_5
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

    def test_from_l2(self):
        expected_p0_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        expected_p1_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        expected_p2_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        expected_p3_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        expected_p4_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        expected_p5_event = PriorityEventL2(data_quality=np.array([]), energy_range=np.array([]),
                                            multi_flag=np.array([]),
                                            spin_angle=np.array([]), number_of_events=np.array([]), ssd_id=np.array([]),
                                            ssd_energy=np.array([]), type=np.array([]), spin_number=np.array([]),
                                            time_of_flight=np.array([]))

        l2_data = CodiceL2HiData(epochs=np.array([]),
                                 priority_event_0=expected_p0_event,
                                 priority_event_1=expected_p1_event,
                                 priority_event_2=expected_p2_event,
                                 priority_event_3=expected_p3_event,
                                 priority_event_4=expected_p4_event,
                                 priority_event_5=expected_p5_event)

        expected_energy_per_nuc = np.arange(72).reshape(6, 2, 2, 3)
        expected_estimated_mass = np.arange(72).reshape(6, 2, 2, 3) * 10

        l3_data_product = (CodiceL3HiDirectEventsBuilder(l2_data)
                           .updated_priority_event_0(energy_per_nuc_with_bounds=expected_energy_per_nuc[0],
                                                     estimated_mass_with_bounds=expected_estimated_mass[0])
                           .updated_priority_event_1(energy_per_nuc_with_bounds=expected_energy_per_nuc[1],
                                                     estimated_mass_with_bounds=expected_estimated_mass[1])
                           .updated_priority_event_2(energy_per_nuc_with_bounds=expected_energy_per_nuc[2],
                                                     estimated_mass_with_bounds=expected_estimated_mass[2])
                           .updated_priority_event_3(energy_per_nuc_with_bounds=expected_energy_per_nuc[3],
                                                     estimated_mass_with_bounds=expected_estimated_mass[3])
                           .updated_priority_event_4(energy_per_nuc_with_bounds=expected_energy_per_nuc[4],
                                                     estimated_mass_with_bounds=expected_estimated_mass[4])
                           .updated_priority_event_5(energy_per_nuc_with_bounds=expected_energy_per_nuc[5],
                                                     estimated_mass_with_bounds=expected_estimated_mass[5])
                           .convert())
        # @formatter:off
        for index, priority_event in enumerate(l2_data.priority_events):
            np.testing.assert_array_equal(priority_event.data_quality, getattr(l3_data_product, f'p{index}_data_quality'))
            np.testing.assert_array_equal(priority_event.energy_range, getattr(l3_data_product, f'p{index}_erge'))
            np.testing.assert_array_equal(priority_event.multi_flag, getattr(l3_data_product, f'p{index}_multi_flag'))
            np.testing.assert_array_equal(priority_event.number_of_events, getattr(l3_data_product, f'p{index}_num_of_events'))
            np.testing.assert_array_equal(priority_event.ssd_energy, getattr(l3_data_product, f'p{index}_ssd_energy'))
            np.testing.assert_array_equal(priority_event.ssd_id, getattr(l3_data_product, f'p{index}_ssd_id'))
            np.testing.assert_array_equal(priority_event.spin_angle, getattr(l3_data_product, f'p{index}_spin_angle'))
            np.testing.assert_array_equal(priority_event.spin_number, getattr(l3_data_product, f'p{index}_spin_number'))
            np.testing.assert_array_equal(priority_event.time_of_flight, getattr(l3_data_product, f'p{index}_tof'))
            np.testing.assert_array_equal(priority_event.type, getattr(l3_data_product, f'p{index}_type'))
            np.testing.assert_array_equal(expected_energy_per_nuc[index, :, :, 0], getattr(l3_data_product, f'p{index}_energy_per_nuc_lower'))
            np.testing.assert_array_equal(expected_energy_per_nuc[index, :, :, 1], getattr(l3_data_product, f'p{index}_energy_per_nuc'))
            np.testing.assert_array_equal(expected_energy_per_nuc[index, :, :, 2], getattr(l3_data_product, f'p{index}_energy_per_nuc_upper'))
            np.testing.assert_array_equal(expected_estimated_mass[index, :, :, 0], getattr(l3_data_product, f'p{index}_estimated_mass_lower'))
            np.testing.assert_array_equal(expected_estimated_mass[index, :, :, 1], getattr(l3_data_product, f'p{index}_estimated_mass'))
            np.testing.assert_array_equal(expected_estimated_mass[index, :, :, 2], getattr(l3_data_product, f'p{index}_estimated_mass_upper'))

    def test_to_codice_hi_l3_direct_events_to_data_product(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2000, 1, 1)])
        kwarg_codice_data_products = {
            "input_metadata": Mock(),
            "epoch": epoch,
            "p0_data_quality": rng.random((len(epoch))),
            "p0_erge": rng.random((len(epoch), 10000)),
            "p0_multi_flag": rng.random((len(epoch), 10000)),
            "p0_num_of_events": rng.random((len(epoch))),
            "p0_ssd_energy": rng.random((len(epoch), 10000)),
            "p0_ssd_id": rng.random((len(epoch), 10000)),
            "p0_spin_angle": rng.random((len(epoch), 10000)),
            "p0_spin_number": rng.random((len(epoch), 10000)),
            "p0_tof": rng.random((len(epoch), 10000)),
            "p0_type": rng.random((len(epoch), 10000)),
            "p0_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p0_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p0_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p0_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p0_estimated_mass": rng.random((len(epoch), 10000)),
            "p0_estimated_mass_upper": rng.random((len(epoch), 10000)),
            "p1_data_quality": rng.random((len(epoch))),
            "p1_erge": rng.random((len(epoch), 10000)),
            "p1_multi_flag": rng.random((len(epoch), 10000)),
            "p1_num_of_events": rng.random((len(epoch))),
            "p1_ssd_energy": rng.random((len(epoch), 10000)),
            "p1_ssd_id": rng.random((len(epoch), 10000)),
            "p1_spin_angle": rng.random((len(epoch), 10000)),
            "p1_spin_number": rng.random((len(epoch), 10000)),
            "p1_tof": rng.random((len(epoch), 10000)),
            "p1_type": rng.random((len(epoch), 10000)),
            "p1_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p1_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p1_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p1_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p1_estimated_mass": rng.random((len(epoch), 10000)),
            "p1_estimated_mass_upper": rng.random((len(epoch), 10000)),
            "p2_data_quality": rng.random((len(epoch))),
            "p2_erge": rng.random((len(epoch), 10000)),
            "p2_multi_flag": rng.random((len(epoch), 10000)),
            "p2_num_of_events": rng.random((len(epoch))),
            "p2_ssd_energy": rng.random((len(epoch), 10000)),
            "p2_ssd_id": rng.random((len(epoch), 10000)),
            "p2_spin_angle": rng.random((len(epoch), 10000)),
            "p2_spin_number": rng.random((len(epoch), 10000)),
            "p2_tof": rng.random((len(epoch), 10000)),
            "p2_type": rng.random((len(epoch), 10000)),
            "p2_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p2_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p2_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p2_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p2_estimated_mass": rng.random((len(epoch), 10000)),
            "p2_estimated_mass_upper": rng.random((len(epoch), 10000)),
            "p3_data_quality": rng.random((len(epoch))),
            "p3_erge": rng.random((len(epoch), 10000)),
            "p3_multi_flag": rng.random((len(epoch), 10000)),
            "p3_num_of_events": rng.random((len(epoch))),
            "p3_ssd_energy": rng.random((len(epoch), 10000)),
            "p3_ssd_id": rng.random((len(epoch), 10000)),
            "p3_spin_angle": rng.random((len(epoch), 10000)),
            "p3_spin_number": rng.random((len(epoch), 10000)),
            "p3_tof": rng.random((len(epoch), 10000)),
            "p3_type": rng.random((len(epoch), 10000)),
            "p3_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p3_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p3_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p3_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p3_estimated_mass": rng.random((len(epoch), 10000)),
            "p3_estimated_mass_upper": rng.random((len(epoch), 10000)),
            "p4_data_quality": rng.random((len(epoch))),
            "p4_erge": rng.random((len(epoch), 10000)),
            "p4_multi_flag": rng.random((len(epoch), 10000)),
            "p4_num_of_events": rng.random((len(epoch))),
            "p4_ssd_energy": rng.random((len(epoch), 10000)),
            "p4_ssd_id": rng.random((len(epoch), 10000)),
            "p4_spin_angle": rng.random((len(epoch), 10000)),
            "p4_spin_number": rng.random((len(epoch), 10000)),
            "p4_tof": rng.random((len(epoch), 10000)),
            "p4_type": rng.random((len(epoch), 10000)),
            "p4_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p4_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p4_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p4_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p4_estimated_mass": rng.random((len(epoch), 10000)),
            "p4_estimated_mass_upper": rng.random((len(epoch), 10000)),
            "p5_data_quality": rng.random((len(epoch))),
            "p5_erge": rng.random((len(epoch), 10000)),
            "p5_multi_flag": rng.random((len(epoch), 10000)),
            "p5_num_of_events": rng.random((len(epoch))),
            "p5_ssd_energy": rng.random((len(epoch), 10000)),
            "p5_ssd_id": rng.random((len(epoch), 10000)),
            "p5_spin_angle": rng.random((len(epoch), 10000)),
            "p5_spin_number": rng.random((len(epoch), 10000)),
            "p5_tof": rng.random((len(epoch), 10000)),
            "p5_type": rng.random((len(epoch), 10000)),
            "p5_energy_per_nuc_lower": rng.random((len(epoch), 10000)),
            "p5_energy_per_nuc": rng.random((len(epoch), 10000)),
            "p5_energy_per_nuc_upper": rng.random((len(epoch), 10000)),
            "p5_estimated_mass_lower": rng.random((len(epoch), 10000)),
            "p5_estimated_mass": rng.random((len(epoch), 10000)),
            "p5_estimated_mass_upper": rng.random((len(epoch), 10000))
        }

        data_product = CodiceL3HiDirectEvents(**kwarg_codice_data_products)
        kwarg_codice_data_products.pop("input_metadata")

        data_product_variables = data_product.to_data_product_variables()
        for data_product_variable, kwarg_codice_data_products in zip(data_product_variables,
                                                                     kwarg_codice_data_products.items()):
            input_name, _ = kwarg_codice_data_products
            self.assertEqual(data_product_variable.name, input_name)
            np.testing.assert_array_equal(getattr(data_product, input_name), data_product_variable.value)

    def test_codice_hi_l3_pitch_angle_to_data_product(self):
        expected_variables = []

        epoch_data = np.array([datetime.now()])
        energy_data = np.array([100, 200])
        pitch_angle = np.array([100, 200])
        gyrophase = np.array([100, 200])
        pitch_angle_size =  len(epoch_data) * len(energy_data) * len(pitch_angle)
        pitch_angle_and_gyrophase_size =  len(epoch_data) * len(energy_data) * len(pitch_angle) * len(gyrophase)

        inputted_data_product_kwargs = {
            "epoch":epoch_data,
            "epoch_delta":np.array([10]),
            "energy":energy_data,
            "energy_delta_plus":np.array([100, 200]),
            "energy_delta_minus":np.array([100, 200]),
            "pitch_angle":pitch_angle,
            "pitch_angle_delta":np.array([100, 200]),
            "gyrophase":gyrophase,
            "gyrophase_delta":np.array([100, 200]),
            "h_intensity_by_pitch_angle":np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle)) + 1,
            "h_intensity_by_pitch_angle_and_gyrophase": np.arange(pitch_angle_and_gyrophase_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) +2,
            "he4_intensity_by_pitch_angle":np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle)),
            "he4_intensity_by_pitch_angle_and_gyrophase":np.arange(pitch_angle_and_gyrophase_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 3,
            "o_intensity_by_pitch_angle":np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle)) + 4,
            "o_intensity_by_pitch_angle_and_gyrophase":np.arange(pitch_angle_and_gyrophase_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 5,
            "fe_intensity_by_pitch_angle":np.arange(pitch_angle_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle)) + 6,
            "fe_intensity_by_pitch_angle_and_gyrophase":np.arange(pitch_angle_and_gyrophase_size).reshape(len(epoch_data), len(energy_data), len(pitch_angle), len(gyrophase)) + 7,
        }

        data_product = CodiceHiL3PitchAngleDataProduct(
            input_metadata=Mock(),
            **inputted_data_product_kwargs
           )
        actual_data_product_variables = data_product.to_data_product_variables()

        for input_variable, actual_data_product_variable in zip(inputted_data_product_kwargs.items(), actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)


    def test_l2_sectored_intensities_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta = np.repeat(len(epoch), 2)
                energy = np.geomspace(2, 1000)
                energy_delta_minus = energy - .4
                energy_delta_plus = energy * 1.6
                spin_sector = np.linspace(0, 360, 24)
                ssd_id = np.linspace(0, 360, 16)
                h_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                he4_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                fe_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta'] = epoch_delta
                cdf_file['energy'] = energy
                cdf_file['energy_delta_minus'] = energy_delta_minus
                cdf_file['energy_delta_plus'] = energy_delta_plus
                cdf_file['spin_sector'] = spin_sector
                cdf_file['ssd_id'] = ssd_id
                cdf_file['h_intensities'] = h_intensities
                cdf_file['he4_intensities'] = he4_intensities
                cdf_file['o_intensities'] = o_intensities
                cdf_file['fe_intensities'] = fe_intensities

            result: CodiceHiL2SectoredIntensitiesData = CodiceHiL2SectoredIntensitiesData.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta, epoch_delta)
            np.testing.assert_array_equal(result.energy, energy)
            np.testing.assert_array_equal(result.energy_delta_minus, energy_delta_minus)
            np.testing.assert_array_equal(result.energy_delta_plus, energy_delta_plus)
            np.testing.assert_array_equal(result.spin_sector, spin_sector)
            np.testing.assert_array_equal(result.ssd_id, ssd_id)
            np.testing.assert_array_equal(result.h_intensities, h_intensities)
            np.testing.assert_array_equal(result.he4_intensities, he4_intensities)
            np.testing.assert_array_equal(result.o_intensities, o_intensities)
            np.testing.assert_array_equal(result.fe_intensities, fe_intensities)



