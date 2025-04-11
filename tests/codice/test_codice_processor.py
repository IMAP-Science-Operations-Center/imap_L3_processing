import unittest
from datetime import datetime
from unittest.mock import patch, sentinel, Mock

import numpy as np

from imap_l3_processing.codice.codice_processor import CodiceProcessor
from imap_l3_processing.codice.l3.direct_event.codice_l3_dependencies import CodiceL3Dependencies
from imap_l3_processing.codice.l3.direct_event.science.tof_lookup import TOFLookup, EnergyPerNuc
from imap_l3_processing.codice.models import PriorityEventL2, CodiceL2HiData
from imap_l3_processing.models import InputMetadata


class TestCodiceProcessor(unittest.TestCase):

    @patch("imap_l3_processing.codice.codice_processor.CodiceL3Dependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.codice_processor.CodiceProcessor.process_l3a")
    def test_process_l3a(self, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l3a", start_date, end_date, 'v02')

        processor = CodiceProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_called_with(sentinel.processing_input_collection)
        mock_process_l3a.assert_called_with(mock_fetch_dependencies.return_value)

    @patch("imap_l3_processing.codice.codice_processor.CodiceL3Dependencies.fetch_dependencies")
    @patch("imap_l3_processing.codice.codice_processor.CodiceProcessor.process_l3a")
    def test_ignores_non_l3_input_metadata(self, mock_process_l3a, mock_fetch_dependencies):
        start_date = datetime(2024, 10, 7, 10, 00, 00)
        end_date = datetime(2024, 10, 8, 10, 00, 00)
        input_metadata = InputMetadata('codice', "l2a", start_date, end_date, 'v02')

        processor = CodiceProcessor(sentinel.processing_input_collection, input_metadata)
        processor.process()

        mock_fetch_dependencies.assert_not_called()
        mock_process_l3a.assert_not_called()

    def test_process_l3a_returns(self):
        epochs = np.array([datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 1, 0, 0, 0)])
        energies = np.array([[2, 4], [6, 8]])
        l2_data = CodiceL2HiData(epochs, *self.create_priority_events(energies))
        energy_per_nuc_dictionary = {i: EnergyPerNuc(i * 10, i * 100, i * 1000) for i in np.arange(1, 25)}
        tof_lookup = TOFLookup(energy_per_nuc_dictionary)
        dependencies = CodiceL3Dependencies(tof_lookup=tof_lookup, codice_l2_hi_data=l2_data)

        processor = CodiceProcessor(Mock(), Mock())
        codice_direct_event_product = processor.process_l3a(dependencies)

        p0_expected_energy_per_nuc = np.array([[10, 20], [30, 40]])
        p1_expected_energy_per_nuc = np.array([[50, 60], [70, 80]])
        p2_expected_energy_per_nuc = np.array([[90, 100], [110, 120]])
        p3_expected_energy_per_nuc = np.array([[130, 140], [150, 160]])
        p4_expected_energy_per_nuc = np.array([[170, 180], [190, 200]])
        p5_expected_energy_per_nuc = np.array([[210, 220], [230, 240]])

        # @formatter: off
        p0_expected_energy_per_nuc = (
            p0_expected_energy_per_nuc, p0_expected_energy_per_nuc * 10, p0_expected_energy_per_nuc * 100)
        p1_expected_energy_per_nuc = (
            p1_expected_energy_per_nuc, p1_expected_energy_per_nuc * 10, p1_expected_energy_per_nuc * 100)
        p2_expected_energy_per_nuc = (
            p2_expected_energy_per_nuc, p2_expected_energy_per_nuc * 10, p2_expected_energy_per_nuc * 100)
        p3_expected_energy_per_nuc = (
            p3_expected_energy_per_nuc, p3_expected_energy_per_nuc * 10, p3_expected_energy_per_nuc * 100)
        p4_expected_energy_per_nuc = (
            p4_expected_energy_per_nuc, p4_expected_energy_per_nuc * 10, p4_expected_energy_per_nuc * 100)
        p5_expected_energy_per_nuc = (
            p5_expected_energy_per_nuc, p5_expected_energy_per_nuc * 10, p5_expected_energy_per_nuc * 100)

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_0,
                                    actual_calculated_mass_lower=codice_direct_event_product.p0_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p0_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p0_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p0_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p0_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p0_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p0_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p0_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p0_expected_energy_per_nuc[2])

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_1,
                                    actual_calculated_mass_lower=codice_direct_event_product.p1_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p1_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p1_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p1_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p1_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p1_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p1_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p1_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p1_expected_energy_per_nuc[2])

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_2,
                                    actual_calculated_mass_lower=codice_direct_event_product.p2_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p2_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p2_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p2_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p2_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p2_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p2_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p2_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p2_expected_energy_per_nuc[2])

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_3,
                                    actual_calculated_mass_lower=codice_direct_event_product.p3_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p3_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p3_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p3_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p3_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p3_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p3_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p3_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p3_expected_energy_per_nuc[2])

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_4,
                                    actual_calculated_mass_lower=codice_direct_event_product.p4_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p4_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p4_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p4_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p4_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p4_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p4_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p4_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p4_expected_energy_per_nuc[2])

        self._assert_estimated_mass(l2_priority_event=l2_data.priority_event_5,
                                    actual_calculated_mass_lower=codice_direct_event_product.p5_estimated_mass_lower,
                                    actual_calculated_mass=codice_direct_event_product.p5_estimated_mass,
                                    actual_calculated_mass_upper=codice_direct_event_product.p5_estimated_mass_upper,
                                    actual_energy_per_nuc_lower=codice_direct_event_product.p5_energy_per_nuc_lower,
                                    actual_energy_per_nuc=codice_direct_event_product.p5_energy_per_nuc,
                                    actual_energy_per_nuc_upper=codice_direct_event_product.p5_energy_per_nuc_upper,
                                    expected_energy_per_nuc_lower=p5_expected_energy_per_nuc[0],
                                    expected_energy_per_nuc=p5_expected_energy_per_nuc[1],
                                    expected_energy_per_nuc_upper=p5_expected_energy_per_nuc[2])

    def create_priority_events(self, energies):
        ssd_id_all_events = np.arange(1, 25).reshape(6, 2, 2) * 1000
        tof_all_events = np.arange(1, 25).reshape(6, 2, 2)

        events = []
        for i in range(6):
            data_quality = np.arange(2) * 1
            energy_range = np.arange(4).reshape(2, 2) + i
            multi_flag = np.arange(4).reshape(2, 2) + i
            number_of_events = np.arange(2) + i
            ssd_energy = ssd_id_all_events[i]
            ssd_id = ssd_id_all_events[i]
            spin_angle = np.arange(4).reshape(2, 2) * i
            spin_number = np.arange(4).reshape(2, 2) * i
            time_of_flight = tof_all_events[i]
            type = np.arange(4).reshape(2, 2) * i
            events.append(PriorityEventL2(data_quality, energy_range, multi_flag, number_of_events, ssd_energy, ssd_id,
                                          spin_angle, spin_number, time_of_flight, type))

        return events

    def _assert_estimated_mass(self, l2_priority_event,
                               actual_calculated_mass_lower, actual_calculated_mass, actual_calculated_mass_upper,
                               actual_energy_per_nuc_lower, actual_energy_per_nuc, actual_energy_per_nuc_upper,
                               expected_energy_per_nuc_lower, expected_energy_per_nuc, expected_energy_per_nuc_upper):
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc_lower,
                                      actual_calculated_mass_lower)
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc, actual_calculated_mass)
        np.testing.assert_array_equal(l2_priority_event.ssd_energy / expected_energy_per_nuc_upper,
                                      actual_calculated_mass_upper)

        np.testing.assert_array_equal(expected_energy_per_nuc_lower, actual_energy_per_nuc_lower)
        np.testing.assert_array_equal(expected_energy_per_nuc, actual_energy_per_nuc)
        np.testing.assert_array_equal(expected_energy_per_nuc_upper, actual_energy_per_nuc_upper)
