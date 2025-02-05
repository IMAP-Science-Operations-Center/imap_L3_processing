import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bitstring import BitStream

from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange


@dataclass
class PHAWord:
    adc_overflow: bool
    adc_value: int
    detector_address: int
    is_high_gain: bool
    is_last_pha: bool


@dataclass
class PHAEvent:
    detectors: list[str]
    adc_values: list[float]
    range: DetectedRange


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
            e_delta_index = event_bitstream.read("uint:9")
            detector_flags = event_bitstream.read("uint:8")
            extended_header = PHAExtendedHeader(detector_flags, e_delta_index, e_prime_index)

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
            is_high_gain = event_bitstream.read("uint:1")
            adc_detector_address = event_bitstream.read("uint:6")
            adc_overflow = event_bitstream.read("uint:1")
            adc_value = event_bitstream.read("uint:11")
            reading_pha = not is_last_event

            pha_word = PHAWord(adc_overflow=adc_overflow, adc_value=adc_value, detector_address=adc_detector_address,
                               is_last_pha=is_last_event,
                               is_high_gain=is_high_gain)
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
    def preprocess(cls, raw_pha_events: [RawPHAEvent]):
        detector_mapping = {}
        with open(Path(__file__).parent / "address_to_detector.csv") as detector_mapping_file:
            reader = csv.reader(detector_mapping_file)
            next(reader)
            for name, address, _ in reader:
                detector_mapping[int(address)] = name

        pha_events = []
        for pha_event in raw_pha_events:
            detectors = []
            adc_values = []
            for word in pha_event.pha_words:
                detectors.append(detector_mapping[word.detector_address])
                adc_values.append(word.adc_value)

            detector_levels = set([d[:2] for d in detectors])
            detected_range = None
            if set(detector_levels) == {"L1", "L2"}:
                detected_range = DetectedRange.R2
            elif set(detector_levels) == {"L1", "L2", "L3"}:
                detected_range = DetectedRange.R3
            elif set(detector_levels) == {"L1", "L2", "L3", "L3"}:
                detected_range = DetectedRange.R4

            pha_events.append(PHAEvent(range=detected_range, detectors=detectors, adc_values=adc_values))

        return pha_events

    @classmethod
    def read_all_pha_events(cls, binary_pha_events: str) -> list[RawPHAEvent]:
        bitstream = BitStream("0b" + binary_pha_events)
        _ = bitstream.read("uint:16")

        events = []
        while (bitstream.len - bitstream.pos >= 48) and bitstream.peek("uint:48") != 0:
            try:
                events.append(cls.read_pha_event(bitstream))
            except Exception as e:
                print("failed to read event")

        return events
