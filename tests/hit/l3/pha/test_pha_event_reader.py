import csv
import unittest

from bitstring import BitArray, BitStream
from spacepy.pycdf import CDF

from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAEventReader, RawPHAEvent, PHAWord, Detector
from tests.hit.l3.hit_test_builders import create_raw_pha_event
from tests.test_helpers import get_test_data_path


class TestPHAEventReader(unittest.TestCase):

    def test_can_create_detector_from_address(self):
        address_to_expected_detector = [
            (35, "L1B1a", Detector(layer=1, side="B", segment="1a", address=35, group="L1B14")),
            (6, "L2A3", Detector(layer=2, side="A", segment="3", address=6, group="L2A")),
            (31, "L4Ao", Detector(layer=4, side="A", segment="o", address=31, group="L4oA")),
            (10, "L4AUnknown_10", Detector(layer=4, side="A", segment="UNKNOWN_10", address=10, group="Unknown")),
        ]

        for address, expected_detector_name, expected_detector in address_to_expected_detector:
            with self.subTest(f"{address} should become {expected_detector_name}"):
                detector = Detector.from_address(address)
                self.assertEqual(expected_detector, detector)

    def test_read_l1_pha_data(self):
        expected_particle_id = 3
        particle_id_binary = "00000011"
        expected_priority_number = 5
        priority_number_binary = "00101"
        expected_stim_tag = False
        stim_tag_binary = "0"
        expected_haz_tag = False
        haz_tag_binary = "0"
        expected_time_tag = 13
        time_tag_binary = "1101"
        expected_a_b_side_flag = False
        a_b_side_binary = "0"
        expected_has_unread_adcs = True
        has_unread_adcs_binary = "1"
        expected_long_event_flag = False
        long_event_flag_binary = "0"
        expected_culling_flag = True
        culling_flag_binary = "1"
        expected_spare = False
        spare_binary = "0"

        pha1_adc_value = 2047
        pha1_adc_value_binary = "11111111111"
        pha1_adc_overflow = False
        pha1_adc_overflow_binary = "0"
        pha1_adc_detector_address = 8
        pha1_detector_address_binary = "001000"
        pha1_is_low_gain = False
        pha1_is_low_gain_binary = "0"
        pha1_is_last_event = False
        pha1_is_last_pha_event = "0"

        pha2_adc_value = 1718
        pha2_adc_value_binary = "11010110110"
        pha2_adc_overflow = True
        pha2_adc_overflow_binary = "1"
        pha2_adc_detector_address = 31
        pha2_detector_address_binary = "011111"
        pha2_is_low_gain = True
        pha2_is_low_gain_binary = "1"
        pha2_is_last_event = True
        pha2_is_last_pha_event = "1"

        event_record_header_parts = [particle_id_binary, priority_number_binary, stim_tag_binary, haz_tag_binary,
                                     time_tag_binary, a_b_side_binary, has_unread_adcs_binary, long_event_flag_binary,
                                     culling_flag_binary, spare_binary]
        pha1_data_parts = [pha1_adc_value_binary, pha1_adc_overflow_binary, pha1_detector_address_binary,
                           pha1_is_low_gain_binary,
                           pha1_is_last_pha_event]
        pha2_data_parts = [pha2_adc_value_binary, pha2_adc_overflow_binary, pha2_detector_address_binary,
                           pha2_is_low_gain_binary,
                           pha2_is_last_pha_event]

        binary = "".join(event_record_header_parts[::-1]) + "".join(pha1_data_parts[::-1]) + "".join(
            pha2_data_parts[::-1])

        bitstream = BitStream("0b" + binary)
        event_data = PHAEventReader.read_pha_event(bitstream)

        self.assertEqual(expected_particle_id, event_data.particle_id)
        self.assertEqual(expected_priority_number, event_data.priority_buffer_num)
        self.assertEqual(expected_stim_tag, event_data.stim_tag)
        self.assertEqual(expected_haz_tag, event_data.haz_tag)
        self.assertEqual(expected_time_tag, event_data.time_tag)
        self.assertEqual(expected_a_b_side_flag, event_data.a_b_side_flag)
        self.assertEqual(expected_has_unread_adcs, event_data.has_unread_adcs)
        self.assertEqual(expected_long_event_flag, event_data.long_event_flag)
        self.assertEqual(expected_culling_flag, event_data.culling_flag)
        self.assertEqual(expected_spare, event_data.spare)
        self.assertIsNone(event_data.extended_header)

        self.assertEqual(2, len(event_data.pha_words))
        self.assertEqual(pha1_adc_value, event_data.pha_words[0].adc_value)
        self.assertEqual(pha1_adc_overflow, event_data.pha_words[0].adc_overflow)
        self.assertEqual(pha1_adc_detector_address, event_data.pha_words[0].detector.address)
        self.assertEqual(pha1_is_low_gain, event_data.pha_words[0].is_low_gain)
        self.assertEqual(pha1_is_last_event, event_data.pha_words[0].is_last_pha)

        self.assertEqual(pha2_adc_value, event_data.pha_words[1].adc_value)
        self.assertEqual(pha2_adc_overflow, event_data.pha_words[1].adc_overflow)
        self.assertEqual(pha2_adc_detector_address, event_data.pha_words[1].detector.address)
        self.assertEqual(pha2_is_low_gain, event_data.pha_words[1].is_low_gain)
        self.assertEqual(pha2_is_last_event, event_data.pha_words[1].is_last_pha)

    def test_reads_extended_header_section_when_long_event_flag_set(self):
        header_binary = self._create_pha_header(long_event_flag=True)
        expected_detector_flags = 34
        expected_delta_e_index = 108
        expected_e_prime_index = 105

        extended_header_binary = "".join([
            BitArray(uint=expected_e_prime_index, length=7).bin,
            BitArray(uint=expected_delta_e_index, length=9).bin,
            BitArray(uint=expected_detector_flags, length=8).bin,
        ])

        pha_word1 = self._create_pha_data(is_last_event=False)
        pha_word2 = self._create_pha_data(is_last_event=True)
        bitstream = BitStream("0b" + header_binary + extended_header_binary + pha_word1 + pha_word2)
        event_data = PHAEventReader.read_pha_event(bitstream)

        self.assertIsNone(event_data.extended_stim_header)
        self.assertEqual(expected_detector_flags, event_data.extended_header.detector_flags)
        self.assertEqual(expected_delta_e_index, event_data.extended_header.delta_e_index)
        self.assertEqual(expected_e_prime_index, event_data.extended_header.e_prime_index)

    def test_reads_extended_stim_header_section_when_long_event_and_stim_flag_set(self):
        header_binary = self._create_pha_header(stim_tag=True, long_event_flag=True)
        expected_dac_value = 108
        expected_tbd = 105

        expected_stim_step = 12
        expected_stim_gain = False
        expected_a_l_stim = 0
        stim_block_binary = "".join([
            BitArray(uint=expected_a_l_stim, length=1).bin,
            BitArray(uint=0, length=2).bin,
            BitArray(uint=expected_stim_gain, length=1).bin,
            BitArray(uint=expected_stim_step, length=4).bin,
        ])

        extended_stim_header_binary = "".join([
            BitArray(uint=expected_tbd, length=12).bin,
            BitArray(uint=expected_dac_value, length=12).bin,
        ])

        pha_word1 = self._create_pha_data(is_last_event=False)
        pha_word2 = self._create_pha_data(is_last_event=True)
        bitstream = BitStream(
            "0b" + header_binary + extended_stim_header_binary + stim_block_binary + pha_word1 + pha_word2)
        event_data = PHAEventReader.read_pha_event(bitstream)

        self.assertIsNone(event_data.extended_header)

        self.assertEqual(expected_stim_step, event_data.stim_block.stim_step)
        self.assertEqual(expected_stim_gain, event_data.stim_block.stim_gain)
        self.assertEqual(expected_a_l_stim, event_data.stim_block.a_l_stim)

        self.assertEqual(expected_dac_value, event_data.extended_stim_header.dac_value)

    def test_reads_stim_block_when_stim_flag_set(self):
        header_binary = self._create_pha_header(stim_tag=True, long_event_flag=False)

        expected_stim_step = 11
        expected_stim_gain = True
        expected_a_l_stim = True
        stim_block_binary = "".join([
            BitArray(uint=expected_a_l_stim, length=1).bin,
            BitArray(uint=0, length=2).bin,
            BitArray(uint=expected_stim_gain, length=1).bin,
            BitArray(uint=expected_stim_step, length=4).bin,
        ])

        pha_word1 = self._create_pha_data(is_last_event=False)
        pha_word2 = self._create_pha_data(is_last_event=True)
        bitstream = BitStream("0b" + header_binary + stim_block_binary + pha_word1 + pha_word2)
        event_data = PHAEventReader.read_pha_event(bitstream)

        self.assertIsNone(event_data.extended_header)
        self.assertIsNone(event_data.extended_stim_header)

        self.assertEqual(expected_stim_step, event_data.stim_block.stim_step)
        self.assertEqual(expected_stim_gain, event_data.stim_block.stim_gain)
        self.assertEqual(expected_a_l_stim, event_data.stim_block.a_l_stim)

    def test_read_all_pha_events_stops_reading_events_when_it_finds_all_zeros(self):
        event1 = create_raw_pha_event()
        event2 = create_raw_pha_event()
        event3 = create_raw_pha_event()

        number_of_events = 33715
        event_record_buffer = BitArray(uint=number_of_events, length=16).bin
        event_record_buffer += self._create_pha_event_binary(event1)
        event_record_buffer += self._create_pha_event_binary(event2)
        event_record_buffer += self._create_pha_event_binary(event3)
        event_record_buffer += "0" * 48

        events = PHAEventReader.read_all_pha_events(event_record_buffer)

        self.assertEqual(3, len(events))
        self.assertEqual(event1, events[0])
        self.assertEqual(event2, events[1])
        self.assertEqual(event3, events[2])

    def test_read_events_adds_extra_bits_for_odd_length_pha_words(self):
        event1 = create_raw_pha_event(pha_words=[
            PHAWord(detector=Detector.from_address(14), adc_value=1000, adc_overflow=True, is_low_gain=True,
                    is_last_pha=True)])

        event2 = create_raw_pha_event()

        number_of_events = 33715
        event_record_buffer = BitArray(uint=number_of_events, length=16).bin
        event_record_buffer += self._create_pha_event_binary(event1)
        event_record_buffer += self._create_pha_event_binary(event2)

        events = PHAEventReader.read_all_pha_events(event_record_buffer)

        self.assertEqual(2, len(events))
        self.assertEqual(event1, events[0])
        self.assertEqual(event2, events[1])

    def test_reads_pha_events_from_cdf(self):
        cdf = CDF(str(get_test_data_path("hit/pha_events/imap_hit_l1a_direct-events_20100105_v003.cdf")))

        event_total = 0
        for event_buffer in cdf["pha_raw"][...]:
            events = PHAEventReader.read_all_pha_events(event_buffer)
            event_total += len(events)

        self.assertEqual(event_total, 7050)

    def test_verify_against_HIT_script_output(self):
        bitstream = BitStream(filename=get_test_data_path("hit/pha_events/full_event_record_buffer.bin"))

        events = PHAEventReader.read_all_pha_events(bitstream.bin)

        with open(get_test_data_path("hit/pha_events/expected_pha_data_for_all_events.csv")) as expected_file:
            reader = csv.reader(expected_file)
            for expected_row, event in zip(reader, events):
                row = ["      " for _ in range(128)]
                for pha_word in event.pha_words:
                    if pha_word.adc_overflow:
                        continue
                    if not pha_word.is_low_gain:
                        index = pha_word.detector.address
                        row[index] = "{: 6}".format(pha_word.adc_value)
                    else:
                        index = pha_word.detector.address + 64
                        row[index] = "{: 6}".format(pha_word.adc_value)
                self.assertEqual(expected_row[2:], row)

    def _create_pha_event_binary(self, pha_event: RawPHAEvent) -> str:
        binary_string = ""

        binary_string += self._create_pha_header(particle_id=pha_event.particle_id,
                                                 buf_priority_num=pha_event.priority_buffer_num,
                                                 stim_tag=pha_event.stim_tag, haz_tag=pha_event.haz_tag,
                                                 time_tag=pha_event.time_tag,
                                                 a_b_side=pha_event.a_b_side_flag,
                                                 has_unread_adcs=pha_event.has_unread_adcs,
                                                 long_event_flag=pha_event.long_event_flag,
                                                 culling_flag=pha_event.culling_flag,
                                                 spare=pha_event.spare)

        if pha_event.extended_header is not None:
            binary_string += "".join([
                BitArray(uint=pha_event.extended_header.e_prime_index, length=8).bin,
                BitArray(uint=pha_event.extended_header.delta_e_index, length=8).bin,
                BitArray(uint=pha_event.extended_header.detector_flags, length=8).bin,
            ])

        if pha_event.stim_block is not None:
            binary_string += "".join([
                BitArray(uint=pha_event.stim_block.a_l_stim, length=1).bin,
                BitArray(uint=0, length=2).bin,
                BitArray(uint=pha_event.stim_block.stim_gain, length=1).bin,
                BitArray(uint=pha_event.stim_block.stim_step, length=4).bin,
            ])

        if pha_event.extended_stim_header is not None:
            binary_string += "".join([
                BitArray(uint=pha_event.extended_stim_header.tbd, length=12).bin,
                BitArray(uint=pha_event.extended_stim_header.dac_value, length=12).bin,
            ])

        for word in pha_event.pha_words:
            binary_string += self._create_pha_data(adc_value=word.adc_value, adc_overflow=word.adc_overflow,
                                                   adc_detector_address=word.detector.address,
                                                   is_low_gain=word.is_low_gain, is_last_event=word.is_last_pha)
        if len(pha_event.pha_words) % 2 == 1:
            binary_string += "0000"

        return binary_string

    def _create_pha_header(self,
                           particle_id: int = 0,
                           buf_priority_num: int = 0,
                           stim_tag: bool = False,
                           haz_tag: bool = False,
                           time_tag: int = 0,
                           a_b_side: bool = False,
                           has_unread_adcs: bool = False,
                           long_event_flag: bool = False,
                           culling_flag: bool = False,
                           spare: bool = False):
        bit_arrays = [
            BitArray(uint=int(spare), length=1),
            BitArray(uint=int(culling_flag), length=1),
            BitArray(uint=int(long_event_flag), length=1),
            BitArray(uint=int(has_unread_adcs), length=1),
            BitArray(uint=int(a_b_side), length=1),
            BitArray(uint=time_tag, length=4),
            BitArray(uint=int(haz_tag), length=1),
            BitArray(uint=int(stim_tag), length=1),
            BitArray(uint=buf_priority_num, length=5),
            BitArray(uint=particle_id, length=8),
        ]
        return "".join([bit_array.bin for bit_array in bit_arrays])

    def _create_pha_data(self,
                         adc_value=0,
                         adc_overflow=False,
                         adc_detector_address=0,
                         is_low_gain=True,
                         is_last_event=True,
                         ):
        bit_arrays = [
            BitArray(uint=int(is_last_event), length=1),
            BitArray(uint=int(is_low_gain), length=1),
            BitArray(uint=adc_detector_address, length=6),
            BitArray(uint=adc_overflow, length=1),
            BitArray(uint=adc_value, length=11),
        ]

        return "".join([bit_array.bin for bit_array in bit_arrays])
