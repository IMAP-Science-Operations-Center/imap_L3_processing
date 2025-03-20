from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import clip

from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAWord, RawPHAEvent
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, Detector, \
    CosineCorrectionLookupTable, DetectorRange
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable, DetectorGain
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import HitEventTypeLookup
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup


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


def analyze_event(event: RawPHAEvent, gain_lookup: GainLookupTable, rule_lookup: HitEventTypeLookup) -> Optional[
    EventAnalysis]:
    def calculate_mev_with_bound_lookup(pha_word: PHAWord) -> float:
        return calculate_mev(pha_word, gain_lookup)

    words = [word for word in event.pha_words]
    groups_to_words = {}
    for word in words:
        if word.detector.group in groups_to_words.keys():
            groups_to_words[word.detector.group].append(word)
        else:
            groups_to_words[word.detector.group] = [word]

    rule = rule_lookup.lookup_range(set(groups_to_words.keys()))
    if rule is not None:
        highest_value_words_per_group = {}
        for include_group in rule.included_detector_groups:
            highest_value_words_per_group[include_group] = max(groups_to_words[include_group],
                                                               key=calculate_mev_with_bound_lookup)

        l1_detector = [group for group in rule.included_detector_groups if group[0:3] == f"L1{rule.range.side.name}"][0]
        l2_detector = [group for group in rule.included_detector_groups if group[0:3] == f"L2{rule.range.side.name}"][0]

        if rule.range.range == DetectorRange.R2:
            e_prime_group = l2_detector
            e_delta_group = l1_detector
        elif rule.range.range == DetectorRange.R3:
            e_prime_group = \
                [group for group in rule.included_detector_groups if group[0:3] == f"L3{rule.range.side.name}"][0]
            e_delta_group = l2_detector
        elif rule.range.range == DetectorRange.R4:
            opposite_side = 'A' if rule.range.side.name == 'B' else 'B'
            e_prime_group = [group for group in rule.included_detector_groups if group[0:3] == f"L3{opposite_side}"][0]
            e_delta_group = \
                [group for group in rule.included_detector_groups if group[0:3] == f"L3{rule.range.side.name}"][0]

        return EventAnalysis(range=rule.range,
                             l1_detector=highest_value_words_per_group[l1_detector].detector,
                             l2_detector=highest_value_words_per_group[l2_detector].detector,
                             e_delta_word=highest_value_words_per_group[e_delta_group],
                             e_prime_word=highest_value_words_per_group[e_prime_group],
                             words_with_highest_energy=list(highest_value_words_per_group.values()))


@dataclass
class ValidDetectorRange:
    e_prime_min: float
    e_prime_max: float
    delta_e_min: float
    delta_e_max: float

    def is_in_range(self, e_prime: float, delta_e: float) -> bool:
        return self.e_prime_min <= e_prime <= self.e_prime_max and self.delta_e_min <= delta_e <= self.delta_e_max


valid_ranges = {
    DetectorRange.R2: ValidDetectorRange(0.2, 860, 0.1, 430),
    DetectorRange.R3: ValidDetectorRange(1.0, 4300, 0.2, 860),
    DetectorRange.R4: ValidDetectorRange(1.0, 4300, 1.0, 4300),
}


def compute_charge(detected_range: DetectedRange, delta_e: float, e_prime: float,
                   double_power_law_lookup: RangeFitLookup) -> float:
    if valid_ranges[detected_range.range].is_in_range(e_prime, delta_e):
        charges, deltas = double_power_law_lookup.evaluate_e_prime(detected_range, e_prime)
        assert np.array_equal(deltas, np.sort(deltas)), "values are not increasing"
        index_2 = clip(np.searchsorted(deltas, delta_e), 1, len(charges) - 1)
        index_1 = index_2 - 1

        B = np.log(charges[index_2] / charges[index_1]) / np.log(deltas[index_2] / deltas[index_1])
        A = charges[index_1] / (deltas[index_1] ** B)

        return A * delta_e ** B
    return np.nan


def process_pha_event(event: RawPHAEvent, cosine_table: CosineCorrectionLookupTable, gain_table: GainLookupTable,
                      range_fit_lookup: RangeFitLookup, rule_lookup: HitEventTypeLookup) -> \
        EventOutput:
    event_analysis = analyze_event(event, gain_table, rule_lookup)
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
