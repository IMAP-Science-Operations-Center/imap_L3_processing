import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from bitstring import BitStream


@dataclass
class Detector:
    layer: Literal[1, 2, 3, 4]
    side: Literal["A", "B"]
    segment: str
    address: int
    group: str

    def __str__(self):
        return "L" + str(self.layer) + self.side + self.segment

    detector_mapping = {}

    @classmethod
    def from_address(cls, address: int):
        if cls.detector_mapping == {}:
            with open(Path(__file__).parent / "address_to_detector.csv") as detector_mapping_file:
                reader = csv.reader(detector_mapping_file)
                next(reader)
                for name, address_str, group in reader:
                    cls.detector_mapping[int(address_str)] = (name, group)

        detector_name_and_group = cls.detector_mapping.get(address)

        if detector_name_and_group is not None:
            detector_name, group = detector_name_and_group
            layer = int(detector_name[1])
            return cls(layer=layer,
                       side=detector_name[2],
                       segment=detector_name[3:],
                       address=address,
                       group=group)

        else:
            return cls(layer=4, side="A", segment=f"UNKNOWN_{address}", address=address, group="Unknown")


@dataclass
class PHAWord:
    adc_overflow: bool
    adc_value: int
    detector: Detector
    is_low_gain: bool
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
class RawPHAEvent:
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
    stim_block: Optional[StimBlock] = None
    extended_stim_header: Optional[ExtendedStimHeader] = None


class PHAEventReader:
    @classmethod
    def read_pha_event(cls, event_bitstream: BitStream) -> RawPHAEvent:
        spare = event_bitstream.read("uint:1")
        culling_flag = event_bitstream.read("uint:1")
        long_event_flag = event_bitstream.read("uint:1")
        has_unread_adcs = event_bitstream.read("uint:1")
        a_b_side_flag = event_bitstream.read("uint:1")

        time_tag = event_bitstream.read("uint:4")

        haz_tag = event_bitstream.read("uint:1")
        stim_tag = event_bitstream.read("uint:1")

        priority_buffer_num = event_bitstream.read("uint:5")

        particle_id = event_bitstream.read("uint:8")

        extended_header = None
        extended_stim_header = None
        stim_block = None
        if long_event_flag and not stim_tag:
            e_prime_index = event_bitstream.read("uint:7")
            delta_e_index = event_bitstream.read("uint:9")
            detector_flags = event_bitstream.read("uint:8")
            extended_header = PHAExtendedHeader(detector_flags, delta_e_index, e_prime_index)

        elif stim_tag:
            if long_event_flag:
                tbd = event_bitstream.read("uint:12")
                dac_value = event_bitstream.read("uint:12")
                extended_stim_header = ExtendedStimHeader(dac_value, tbd)

            a_l_stim = event_bitstream.read("uint:1")
            unused = event_bitstream.read("uint:2")
            stim_gain = event_bitstream.read("uint:1")
            stim_step = event_bitstream.read("uint:4")
            stim_block = StimBlock(stim_step=stim_step, stim_gain=stim_gain, unused=unused, a_l_stim=a_l_stim)

        reading_pha = True
        pha_words = []
        while reading_pha:
            is_last_event = event_bitstream.read("uint:1")
            is_low_gain = event_bitstream.read("uint:1")
            adc_detector_address = event_bitstream.read("uint:6")
            adc_overflow = event_bitstream.read("uint:1")
            adc_value = event_bitstream.read("uint:11")
            reading_pha = not is_last_event

            pha_word = PHAWord(adc_overflow=adc_overflow, adc_value=adc_value,
                               detector=Detector.from_address(adc_detector_address),
                               is_last_pha=is_last_event,
                               is_low_gain=is_low_gain)
            pha_words.append(pha_word)

        if len(pha_words) % 2 == 1:
            event_bitstream.read("uint:4")

        return RawPHAEvent(particle_id=particle_id,
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
                           extended_header=extended_header,
                           extended_stim_header=extended_stim_header,
                           stim_block=stim_block
                           )

    @classmethod
    def read_all_pha_events(cls, binary_pha_events: str) -> list[RawPHAEvent]:
        bitstream = BitStream("0b" + binary_pha_events)

        events = []
        while (bitstream.len - bitstream.pos >= 48) and bitstream.peek("uint:48") != 0:
            events.append(cls.read_pha_event(bitstream))

        return events
