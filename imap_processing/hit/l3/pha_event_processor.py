from dataclasses import dataclass
from typing import Optional

from bitstring import BitStream


@dataclass
class PHAWord:
    adc_value: int
    detector_address: int
    is_high_gain: bool
    is_last_pha: bool


@dataclass
class PHAExtendedHeader:
    detector_flags: int
    delta_e_index: int
    e_prime_index: bool


@dataclass
class ExtendedStimHeader:
    dac_value: int
    tbd: int


@dataclass
class StimBlock:
    stim_step: int
    stim_gain: int
    unused: int
    a_l_stim: bool


@dataclass
class PHAEvent:
    particle_id: int
    priority_buffer_num: int
    stim_tag: bool
    haz_tag: bool
    time_tag: int
    a_b_side_flag: bool
    has_unread_adcs: bool
    long_event_flag: bool
    culling_flag: bool
    spare: bool
    pha_words: list[PHAWord]
    extended_header: Optional[PHAExtendedHeader] = None
    stim_block = int


def read_l1_pha_data(pha_event) -> PHAEvent:
    bitstream = BitStream("0b" + pha_event)
    particle_id = bitstream.read("uint:8")
    priority_buffer_num = bitstream.read("uint:5")
    stim_tag = bitstream.read("uint:1")
    haz_tag = bitstream.read("uint:1")
    time_tag = bitstream.read("uint:4")
    a_b_side_flag = bitstream.read("uint:1")
    has_unread_adcs = bitstream.read("uint:1")
    long_event_flag = bitstream.read("uint:1")
    culling_flag = bitstream.read("uint:1")
    spare = bitstream.read("uint:1")

    if stim_tag:
        stim_step = bitstream.read("uint:4")
        stim_gain = bitstream.read("uint:1")
        unused = bitstream.read("uint:2")
        a_l_stim = bitstream.read("uint:1")
        StimBlock(stim_step=stim_step, stim_gain=stim_gain, unused=unused, a_l_stim=a_l_stim)

    extended_header = None
    if long_event_flag:
        detector_flags = bitstream.read("uint:8")
        e_delta_index = bitstream.read("uint:8")
        e_prime_index = bitstream.read("uint:8")
        extended_header = PHAExtendedHeader(detector_flags, e_delta_index, e_prime_index)

    reading_pha = True
    pha_words = []
    while reading_pha:
        adc_value = bitstream.read("uint:12")
        adc_detector_address = bitstream.read("uint:6")
        is_high_gain = bitstream.read("uint:1")
        is_last_event = bitstream.read("uint:1")
        reading_pha = not is_last_event

        pha_word = PHAWord(adc_value=adc_value, detector_address=adc_detector_address, is_last_pha=is_last_event,
                           is_high_gain=is_high_gain)

        pha_words.append(pha_word)

    return PHAEvent(particle_id=particle_id,
                    priority_buffer_num=priority_buffer_num,
                    stim_tag=stim_tag,
                    haz_tag=haz_tag,
                    time_tag=time_tag,
                    a_b_side_flag=a_b_side_flag,
                    has_unread_adcs=has_unread_adcs,
                    long_event_flag=long_event_flag,
                    culling_flag=culling_flag,
                    spare=spare,
                    pha_words=pha_words,
                    extended_header=extended_header
                    )
