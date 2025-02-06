from dataclasses import dataclass
from typing import Optional

from imap_processing.hit.l3.pha.pha_event_reader import PHAWord, RawPHAEvent
from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, Detector
from imap_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable, DetectorGain


@dataclass
class EventAnalysis:
    range: DetectedRange
    l1_detector: Detector
    l2_detector: Detector
    e_delta_word: PHAWord
    e_prime_word: PHAWord


def calculate_mev(word: PHAWord, gain_lookup_table: GainLookupTable) -> float:
    gain_level = DetectorGain.LOW if word.is_low_gain else DetectorGain.HIGH
    gain_coeffs = gain_lookup_table[gain_level][word.detector.address]
    return word.adc_value * gain_coeffs.a + gain_coeffs.b


def analyze_event(event: RawPHAEvent, gain_lookup: GainLookupTable) -> Optional[EventAnalysis]:
    def calculate_mev_with_bound_lookup(pha_word: PHAWord) -> float:
        return calculate_mev(pha_word, gain_lookup)

    l1_sides = {word.detector.side for word in event.pha_words if word.detector.layer == 1}

    if len(l1_sides) != 1:
        return None
    l1_word = max((word for word in event.pha_words if word.detector.layer == 1), key=calculate_mev_with_bound_lookup)
    l1_detector = l1_word.detector

    matching_l2 = [word for word in event.pha_words if
                   word.detector.layer == 2 and word.detector.side == l1_detector.side]
    if len(matching_l2) == 0:
        return None
    l2_word = max((word for word in matching_l2), key=calculate_mev_with_bound_lookup)
    l2_detector = l2_word.detector

    l3_sides = {word.detector.side for word in event.pha_words if word.detector.layer == 3}
    if l3_sides == set():
        detected_range = DetectedRange.R2

        e_delta_word = l1_word
        e_prime_word = l2_word
    elif l3_sides == l1_sides:
        detected_range = DetectedRange.R3

        e_delta_word = l2_word
        e_prime_word = max((word for word in event.pha_words if word.detector.layer == 3),
                           key=calculate_mev_with_bound_lookup)
    elif len(l3_sides) == 2:
        detected_range = DetectedRange.R4

        e_delta_word = max(
            (word for word in event.pha_words if word.detector.layer == 3 and word.detector.side == l1_detector.side),
            key=calculate_mev_with_bound_lookup)
        e_prime_word = max(
            (word for word in event.pha_words if word.detector.layer == 3 and word.detector.side != l1_detector.side),
            key=calculate_mev_with_bound_lookup)
    else:
        return None

    l2_on_other_side = len([word for word in event.pha_words if
                            word.detector.layer == 2 and word.detector.side != l1_detector.side]) > 0

    if detected_range in [DetectedRange.R2, DetectedRange.R3] and l2_on_other_side:
        return None

    return EventAnalysis(range=detected_range,
                         l1_detector=l1_detector,
                         l2_detector=l2_detector,
                         e_delta_word=e_delta_word,
                         e_prime_word=e_prime_word)
