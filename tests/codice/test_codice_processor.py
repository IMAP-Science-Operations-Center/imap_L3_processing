import unittest
from datetime import datetime
from unittest.mock import Mock, call

import numpy as np

from imap_l3_processing.codice.codice_processor import CodiceProcessor
from imap_l3_processing.codice.l2.direct_event.codice_l2_dependencies import CodiceL2Dependencies
from imap_l3_processing.codice.models import CodiceL1aHiData, PriorityEventL1, CodiceL2HiDataProduct
from imap_l3_processing.models import InputMetadata
from tests.test_helpers import NumpyArrayMatcher


class TestCodiceProcessor(unittest.TestCase):
    def test_process_l2a_creates_creates_dataproduct(self):
        mock_energy_lookup = Mock()
        number_of_values = 3
        epochs, p0_event, p1_event, p2_event, p3_event, p4_event, p5_event = self._create_priority_events(
            number_of_values)

        codice_l1a_hi_data = CodiceL1aHiData(
            epochs=epochs,
            priority_event_0=p0_event,
            priority_event_1=p1_event,
            priority_event_2=p2_event,
            priority_event_3=p3_event,
            priority_event_4=p4_event,
            priority_event_5=p5_event
        )

        codice_l2_dependencies = CodiceL2Dependencies(codice_l1a_hi_data=codice_l1a_hi_data,
                                                      energy_lookup_table=mock_energy_lookup,
                                                      time_of_flight_lookup_table=Mock(),
                                                      azimuth_lookup_table=Mock())
        number_of_priority_events = 6

        codice_l2_dependencies.energy_lookup_table.convert_to_mev.side_effect = list(
            np.arange(number_of_values * number_of_priority_events * len(epochs)))

        codice_input_metadata = InputMetadata(instrument="codice", data_level="l2", start_date=None, end_date=None,
                                              version="latest")
        upstream_dependencies = Mock()
        processor = CodiceProcessor([upstream_dependencies], codice_input_metadata)
        data_product: CodiceL2HiDataProduct = processor.process_l2(codice_l2_dependencies)

        # @formatter:off
        energy_lookup_expected_calls = []
        for e in range(len(codice_l1a_hi_data.epochs)):
            for p in range(number_of_priority_events):

                energy_lookup_expected_calls.extend(
                    [
                        call(NumpyArrayMatcher(p0_event.ssd_id[e][p]), NumpyArrayMatcher(p0_event.energy_range[e][p]), NumpyArrayMatcher(p0_event.ssd_energy[e][p])),
                        call(NumpyArrayMatcher(p1_event.ssd_id[e][p]), NumpyArrayMatcher(p1_event.energy_range[e][p]), NumpyArrayMatcher(p1_event.ssd_energy[e][p])),
                        call(NumpyArrayMatcher(p2_event.ssd_id[e][p]), NumpyArrayMatcher(p2_event.energy_range[e][p]), NumpyArrayMatcher(p2_event.ssd_energy[e][p])),
                        call(NumpyArrayMatcher(p3_event.ssd_id[e][p]), NumpyArrayMatcher(p3_event.energy_range[e][p]), NumpyArrayMatcher(p3_event.ssd_energy[e][p])),
                        call(NumpyArrayMatcher(p4_event.ssd_id[e][p]), NumpyArrayMatcher(p4_event.energy_range[e][p]), NumpyArrayMatcher(p4_event.ssd_energy[e][p])),
                        call(NumpyArrayMatcher(p5_event.ssd_id[e][p]), NumpyArrayMatcher(p5_event.energy_range[e][p]), NumpyArrayMatcher(p5_event.ssd_energy[e][p])),
                    ]
                )
        mock_energy_lookup.convert_to_mev.assert_has_calls(energy_lookup_expected_calls)


        for  epoch_dim in range(len(epochs)):
            for  priority_dim in range(6):
                for  value_dim in range(3):
                    expected_converted = epoch_dim + priority_dim + value_dim
                    converted_energy = data_product.energy[epoch_dim][priority_dim][value_dim]
                    self.assertEqual(converted_energy,expected_converted)

    def _create_priority_events(self, number_of_values):
        rng = np.random.default_rng()
        epoch = np.array([datetime(2025, 4, 1), datetime(2025, 4, 2), datetime(2025, 4, 3), ])
        events: list[PriorityEventL1] = []
        for i in range(6):
            data_quality = rng.random(number_of_values)
            energy_range = rng.random((len(epoch), number_of_values))
            multi_flag = rng.random((len(epoch), number_of_values))
            number_of_events = rng.random(number_of_values)
            ssd_energy = rng.random((len(epoch), number_of_values))
            ssd_id = rng.random((len(epoch), number_of_values))
            spin_angle = rng.random((len(epoch), number_of_values))
            spin_number = rng.random((len(epoch), number_of_values))
            time_of_flight = rng.random((len(epoch), number_of_values))
            type = rng.random((len(epoch), number_of_values))

            events.append(PriorityEventL1(data_quality, energy_range, multi_flag, number_of_events, ssd_energy, ssd_id,
                                          spin_angle, spin_number, time_of_flight, type))
        p0, p1, p2, p3, p4, p5 = [e for e in events]
        return epoch, p0, p1, p2, p3, p4, p5
