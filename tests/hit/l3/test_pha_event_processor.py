import unittest

from bitstring import BitArray

from imap_processing.hit.l3 import pha_event_processor


class TestPHAEventProcessor(unittest.TestCase):

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
            BitArray(uint=particle_id, length=8),
            BitArray(uint=buf_priority_num, length=5),
            BitArray(uint=int(stim_tag), length=1),
            BitArray(uint=int(haz_tag), length=1),
            BitArray(uint=time_tag, length=4),
            BitArray(uint=int(a_b_side), length=1),
            BitArray(uint=int(has_unread_adcs), length=1),
            BitArray(uint=int(long_event_flag), length=1),
            BitArray(uint=int(culling_flag), length=1),
            BitArray(uint=int(spare), length=1),
        ]
        return "".join([bit_array.bin for bit_array in bit_arrays])

    def create_pha_word(self,
                        adc_value=0,
                        adc_detector_address=0,
                        is_high_gain=True,
                        is_last_event=True,
                        ):
        bit_arrays = [
            BitArray(uint=adc_value, length=12),
            BitArray(uint=adc_detector_address, length=6),
            BitArray(uint=int(is_high_gain), length=1),
            BitArray(uint=int(is_last_event), length=1),
        ]

        return "".join([bit_array.bin for bit_array in bit_arrays])

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

        pha1_adc_value = 4095
        pha1_adc_value_binary = "111111111111"
        pha1_adc_detector_address = 8
        pha1_detector_address_binary = "001000"
        pha1_is_high_gain = False
        pha1_is_high_gain_binary = "0"
        pha1_is_last_event = False
        pha1_is_last_pha_event = "0"

        pha2_adc_value = 3766
        pha2_adc_value_binary = "111010110110"
        pha2_adc_detector_address = 11
        pha2_detector_address_binary = "001011"
        pha2_is_high_gain = 0
        pha2_is_high_gain_binary = "0"
        pha2_is_last_event = True
        pha2_is_last_pha_event = "1"

        binary = particle_id_binary + priority_number_binary + stim_tag_binary + haz_tag_binary + time_tag_binary \
                 + a_b_side_binary + has_unread_adcs_binary + long_event_flag_binary + culling_flag_binary + spare_binary + \
                 pha1_adc_value_binary + pha1_detector_address_binary + pha1_is_high_gain_binary + pha1_is_last_pha_event + \
                 pha2_adc_value_binary + pha2_detector_address_binary + pha2_is_high_gain_binary + pha2_is_last_pha_event

        event_data = pha_event_processor.read_l1_pha_data(binary)

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
        self.assertEqual(pha1_adc_detector_address, event_data.pha_words[0].detector_address)
        self.assertEqual(pha1_is_high_gain, event_data.pha_words[0].is_high_gain)
        self.assertEqual(pha1_is_last_event, event_data.pha_words[0].is_last_pha)

        self.assertEqual(pha2_adc_value, event_data.pha_words[1].adc_value)
        self.assertEqual(pha2_adc_detector_address, event_data.pha_words[1].detector_address)
        self.assertEqual(pha2_is_high_gain, event_data.pha_words[1].is_high_gain)
        self.assertEqual(pha2_is_last_event, event_data.pha_words[1].is_last_pha)

    def test_reads_extended_header_section_when_long_event_flag_set(self):
        header_binary = self._create_pha_header(long_event_flag=True)
        expected_detector_flags = 34
        expected_delta_e_index = 108
        expected_e_prime_index = 105

        extended_header_binary = "".join([
            BitArray(uint=expected_detector_flags, length=8).bin,
            BitArray(uint=expected_delta_e_index, length=8).bin,
            BitArray(uint=expected_e_prime_index, length=8).bin,
        ])

        pha_word = self.create_pha_word(is_last_event=True)
        event_data = pha_event_processor.read_l1_pha_data(header_binary + extended_header_binary + pha_word)

        self.assertIsNone(event_data.extended_stim_header)
        self.assertEqual(expected_detector_flags, event_data.extended_header.detector_flags)
        self.assertEqual(expected_delta_e_index, event_data.extended_header.delta_e_index)
        self.assertEqual(expected_e_prime_index, event_data.extended_header.e_prime_index)

    def test_reads_extended_stim_header_section_when_long_event_and_stim_flag_set(self):
        header_binary = self._create_pha_header(stim_tag=True, long_event_flag=True)
        expected_stim_block = 34
        expected_dac_value = 108
        expected_tbd = 105

        extended_stim_header_binary = "".join([
            BitArray(uint=expected_stim_block, length=8).bin,
            BitArray(uint=expected_dac_value, length=12).bin,
            BitArray(uint=expected_tbd, length=12).bin,
        ])

        pha_word = self.create_pha_word(is_last_event=True)
        event_data = pha_event_processor.read_l1_pha_data(header_binary + extended_stim_header_binary + pha_word)

        self.assertIsNone(event_data.extended_header)
        self.assertEqual(expected_stim_block, event_data.extended_stim_header.stim_block)
        self.assertEqual(expected_dac_value, event_data.extended_stim_header.dac_value)
        self.assertEqual(expected_tbd, event_data.extended_stim_header.tbd)

    def test_reads_extended_header_section_when_long_event_flag_set(self):
        header_binary = self._create_pha_header(long_event_flag=True)
        expected_detector_flags = 34
        expected_delta_e_index = 108
        expected_e_prime_index = 105

        extended_header_binary = "".join([
            BitArray(uint=expected_detector_flags, length=8).bin,
            BitArray(uint=expected_delta_e_index, length=8).bin,
            BitArray(uint=expected_e_prime_index, length=8).bin,
        ])

        pha_word = self.create_pha_word(is_last_event=True)
        event_data = pha_event_processor.read_l1_pha_data(header_binary + extended_header_binary + pha_word)

        self.assertEqual(expected_detector_flags, event_data.extended_header.detector_flags)
        self.assertEqual(expected_delta_e_index, event_data.extended_header.delta_e_index)
        self.assertEqual(expected_e_prime_index, event_data.extended_header.e_prime_index)
