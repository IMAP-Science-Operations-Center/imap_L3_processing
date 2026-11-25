from dataclasses import dataclass

import numpy as np
from numpy import clip

from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAWord, RawPHAEvent
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import (
    CosineCorrectionLookupTable,
    DetectedRange,
    Detector,
    DetectorRange,
)
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import (
    DetectorGain,
    GainLookupTable,
)
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import (
    HitEventTypeLookup,
)
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from imap_l3_processing.hit.l3.pha.science.window_correction_lookup_table import (
    WindowCorrectionLookupTable,
)


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
    total_energy: float | None
    charge: float | None
    energies: list[float]
    detected_range: DetectedRange | None
    e_delta: float | None
    e_prime: float | None


def calculate_mev(word: PHAWord, gain_lookup_table: GainLookupTable) -> float:
    gain_level = DetectorGain.LOW if word.is_low_gain else DetectorGain.HIGH
    gain_coeffs = gain_lookup_table[gain_level][word.detector.address]
    return word.adc_value * gain_coeffs.a + gain_coeffs.b


def analyze_event(event: RawPHAEvent, rule_lookup: HitEventTypeLookup) -> EventAnalysis | None:

    def get_adc_value(pha_word: PHAWord):
        return 20 * pha_word.adc_value if pha_word.is_low_gain else pha_word.adc_value

    words = [word for word in event.pha_words]
    groups_to_words = {}
    for word in words:
        if word.detector.group in groups_to_words:
            groups_to_words[word.detector.group].append(word)
        else:
            groups_to_words[word.detector.group] = [word]

    rule = rule_lookup.lookup_range(set(groups_to_words.keys()))
    if rule is not None:
        highest_value_words_per_group = {}
        for include_group in rule.included_detector_groups:
            unsaturated_words = [
                word for word in groups_to_words[include_group] if word.adc_value < 2047
            ]
            if len(unsaturated_words) == 0:
                return None
            highest_value_words_per_group[include_group] = max(
                unsaturated_words, key=get_adc_value
            )

        l1_detector = next(
            group
            for group in rule.included_detector_groups
            if group[0:3] == f"L1{rule.range.side.name}"
        )
        l2_detector = next(
            group
            for group in rule.included_detector_groups
            if group[0:3] == f"L2{rule.range.side.name}"
        )

        if rule.range.range == DetectorRange.R2:
            e_prime_group = l2_detector
            e_delta_group = l1_detector
        elif rule.range.range == DetectorRange.R3:
            e_prime_group = next(
                group
                for group in rule.included_detector_groups
                if group[0:3] == f"L3{rule.range.side.name}"
            )
            e_delta_group = l2_detector
        elif rule.range.range == DetectorRange.R4:
            opposite_side = "A" if rule.range.side.name == "B" else "B"
            e_prime_group = next(
                group
                for group in rule.included_detector_groups
                if group[0:3] == f"L3{opposite_side}"
            )
            e_delta_group = next(
                group
                for group in rule.included_detector_groups
                if group[0:3] == f"L3{rule.range.side.name}"
            )

        return EventAnalysis(
            range=rule.range,
            l1_detector=highest_value_words_per_group[l1_detector].detector,
            l2_detector=highest_value_words_per_group[l2_detector].detector,
            e_delta_word=highest_value_words_per_group[e_delta_group],
            e_prime_word=highest_value_words_per_group[e_prime_group],
            words_with_highest_energy=list(highest_value_words_per_group.values()),
        )


def compute_charge(
    detected_range: DetectedRange,
    delta_e: float,
    e_prime: float,
    double_power_law_lookup: RangeFitLookup,
) -> float:
    if delta_e <= 0 or e_prime <= 0:
        return np.nan
    charges, deltas = double_power_law_lookup.evaluate_e_prime(detected_range, e_prime)
    if not np.all(np.diff(deltas) > 0):
        return np.nan
    index_2 = clip(np.searchsorted(deltas, delta_e), 1, len(charges) - 1)
    index_1 = index_2 - 1

    B = np.log(charges[index_2] / charges[index_1]) / np.log(deltas[index_2] / deltas[index_1])
    A = charges[index_1] / (deltas[index_1] ** B)

    return A * delta_e**B


def process_pha_event(
    event: RawPHAEvent,
    cosine_table: CosineCorrectionLookupTable,
    gain_table: GainLookupTable,
    window_correction_table: WindowCorrectionLookupTable,
    range_fit_lookup: RangeFitLookup,
    rule_lookup: HitEventTypeLookup,
) -> EventOutput:
    event_analysis = analyze_event(event, rule_lookup)
    energies = [calculate_mev(word, gain_table) for word in event.pha_words]
    if not event_analysis:
        return EventOutput(
            original_event=event,
            charge=None,
            total_energy=None,
            energies=energies,
            detected_range=None,
            e_delta=None,
            e_prime=None,
        )

    window_corrected_energies = []
    for word in event_analysis.words_with_highest_energy:
        window_correction_factor = window_correction_table.get_correction_factor(
            event_analysis.range.range, word.detector.address
        )
        deposited_energy = calculate_mev(word, gain_table)
        window_corrected_energies.append(window_correction_factor * deposited_energy)
    total_energy = sum(window_corrected_energies)

    cosine_correction = cosine_table.get_cosine_correction(
        event_analysis.range, event_analysis.l1_detector, event_analysis.l2_detector
    )
    e_delta = cosine_correction * calculate_mev(event_analysis.e_delta_word, gain_table)
    e_prime = cosine_correction * calculate_mev(event_analysis.e_prime_word, gain_table)
    charge = compute_charge(event_analysis.range, e_delta, e_prime, range_fit_lookup)

    return EventOutput(
        original_event=event,
        charge=charge,
        total_energy=total_energy,
        energies=energies,
        detected_range=event_analysis.range,
        e_delta=e_delta,
        e_prime=e_prime,
    )
