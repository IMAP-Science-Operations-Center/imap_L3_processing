from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import clip

from imap_processing.hit.l3.pha.pha_event_reader import PHAWord, RawPHAEvent
from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, Detector, \
    CosineCorrectionLookupTable
from imap_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable, DetectorGain
from imap_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup


@dataclass
class EventAnalysis:
    range: DetectedRange
    l1_detector: Detector
    l2_detector: Detector
    e_delta_word: PHAWord
    e_prime_word: PHAWord
    words_with_highest_energy: list[PHAWord]


@dataclass
class EventOutput:
    original_event: RawPHAEvent
    total_energy: Optional[float]
    charge: Optional[float]
    energies: list[float]
    detected_range: Optional[DetectedRange]
    e_delta: Optional[float]
    e_prime: Optional[float]


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
    words_with_highest_energy = [l1_word, l2_word]

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
        words_with_highest_energy.append(e_prime_word)
    elif len(l3_sides) == 2:
        detected_range = DetectedRange.R4

        e_delta_word = max(
            (word for word in event.pha_words if word.detector.layer == 3 and word.detector.side == l1_detector.side),
            key=calculate_mev_with_bound_lookup)
        e_prime_word = max(
            (word for word in event.pha_words if word.detector.layer == 3 and word.detector.side != l1_detector.side),
            key=calculate_mev_with_bound_lookup)
        words_with_highest_energy.append(e_delta_word)
        words_with_highest_energy.append(e_prime_word)
    else:
        return None

    l2_on_other_side = max(
        (word for word in event.pha_words if word.detector.layer == 2 and word.detector.side != l1_detector.side),
        key=calculate_mev_with_bound_lookup, default=None)

    if detected_range in [DetectedRange.R2, DetectedRange.R3] and l2_on_other_side is not None:
        return None
    elif l2_on_other_side is not None:
        words_with_highest_energy.append(l2_on_other_side)

    return EventAnalysis(range=detected_range,
                         l1_detector=l1_detector,
                         l2_detector=l2_detector,
                         e_delta_word=e_delta_word,
                         e_prime_word=e_prime_word,
                         words_with_highest_energy=words_with_highest_energy)


def compute_charge(detected_range: DetectedRange, delta_e: float, e_prime: float,
                   double_power_law_lookup: RangeFitLookup) -> float:
    charges, deltas = double_power_law_lookup.evaluate_e_prime(detected_range, e_prime)
    assert np.array_equal(deltas, np.sort(deltas)), "values are not increasing"
    index_2 = clip(np.searchsorted(deltas, delta_e), 1, len(charges) - 1)
    index_1 = index_2 - 1

    B = np.log(charges[index_2] / charges[index_1]) / np.log(deltas[index_2] / deltas[index_1])
    A = charges[index_1] / (deltas[index_1] ** B)

    return A * delta_e ** B


def process_pha_event(event: RawPHAEvent, cosine_table: CosineCorrectionLookupTable, gain_table: GainLookupTable,
                      range_fit_lookup: RangeFitLookup) -> \
        EventOutput:
    event_analysis = analyze_event(event, gain_table)
    if event_analysis:
        correction = cosine_table.get_cosine_correction(event_analysis.range, event_analysis.l1_detector,
                                                        event_analysis.l2_detector)

        def calculate_corrected_energy(word):
            return correction * calculate_mev(word, gain_table)

        energies = [calculate_corrected_energy(word) for word in event.pha_words]
        total_energy = sum(
            calculate_corrected_energy(word) for word in event_analysis.words_with_highest_energy)
        e_delta = calculate_corrected_energy(event_analysis.e_delta_word)
        e_prime = calculate_corrected_energy(event_analysis.e_prime_word)
        charge = compute_charge(event_analysis.range, e_delta, e_prime, range_fit_lookup)

        return EventOutput(original_event=event, charge=charge, total_energy=total_energy, energies=energies,
                           detected_range=event_analysis.range, e_delta=e_delta, e_prime=e_prime)
    else:
        energies = [calculate_mev(word, gain_table) for word in event.pha_words]
        return EventOutput(original_event=event, charge=None, total_energy=None, energies=energies, detected_range=None,
                           e_delta=None, e_prime=None)
