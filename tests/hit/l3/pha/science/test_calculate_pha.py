import math
import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np

from imap_l3_processing.hit.l3.pha.pha_event_reader import Detector, PHAWord
from imap_l3_processing.hit.l3.pha.science.calculate_pha import (
    EventAnalysis,
    EventOutput,
    analyze_event,
    calculate_mev,
    compute_charge,
    process_pha_event,
)
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import (
    DetectedRange,
    DetectorRange,
    DetectorSide,
)
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import DetectorGain, Gain
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import Rule
from imap_l3_processing.hit.l3.pha.science.window_correction_lookup_table import (
    WindowCorrectionLookupTable,
)
from tests.hit.l3.hit_test_builders import create_raw_pha_event

MODULE = "imap_l3_processing.hit.l3.pha.science.calculate_pha"


class TestCalculatePHA(unittest.TestCase):
    def test_calculate_mev(self):
        low_gain_a_value = 10.3
        low_gain_b_value = -25.4
        low_gain_adc_value = 136

        high_gain_adc_value = 8
        high_gain_a_value = -5.4
        high_gain_b_value = 100.2

        gain_lookup_table = {
            DetectorGain.LOW: {10: Gain(a=low_gain_a_value, b=low_gain_b_value)},
            DetectorGain.HIGH: {22: Gain(a=high_gain_a_value, b=high_gain_b_value)},
        }
        low_gain_word = PHAWord(
            adc_value=low_gain_adc_value,
            is_low_gain=True,
            detector=Detector.from_address(10),
            is_last_pha=False,
            adc_overflow=False,
        )
        mev = calculate_mev(low_gain_word, gain_lookup_table)
        self.assertEqual(low_gain_a_value * low_gain_adc_value + low_gain_b_value, mev)

        low_gain_word = PHAWord(
            adc_value=high_gain_adc_value,
            is_low_gain=False,
            detector=Detector.from_address(22),
            is_last_pha=False,
            adc_overflow=False,
        )
        mev = calculate_mev(low_gain_word, gain_lookup_table)
        self.assertEqual(high_gain_a_value * high_gain_adc_value + high_gain_b_value, mev)

    def test_analyze_range(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")

        detector_to_pha_value = [
            (detector_1_in_group_1, 10),
            (detector_2_in_group_1, 20),
            (detector_1_in_group_2, 20),
            (detector_1_in_group_3, 10),
        ]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(
            detector_to_pha_value
        )
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(
            range=DetectedRange(DetectorRange.R2, DetectorSide.A),
            included_detector_groups=["L1A14", "L2A"],
            excluded_detector_groups=["L2B"],
        )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A"})

        expected_highest_words = [
            detector_to_word_tuples[str(detector_2_in_group_1)],
            detector_to_word_tuples[str(detector_1_in_group_2)],
        ]
        expected_event_analysis = EventAnalysis(
            range=rule_stub.range,
            l1_detector=detector_2_in_group_1,
            l2_detector=detector_1_in_group_2,
            e_delta_word=detector_to_word_tuples[str(detector_2_in_group_1)],
            e_prime_word=detector_to_word_tuples[str(detector_1_in_group_2)],
            words_with_highest_energy=expected_highest_words,
        )

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_range_mixed_sides(self):
        detector_2_in_L2A_group = Detector(layer=2, side="A", segment="c", address=11, group="L2A")
        detector_1_in_L2A_group = Detector(layer=2, side="A", segment="b", address=10, group="L2A")
        detector_1_in_L2B_group = Detector(layer=2, side="B", segment="c", address=15, group="L2B")
        detector_1_in_L1B14_group = Detector(
            layer=1, side="B", segment="a", address=20, group="L1B14"
        )

        detector_to_pha_value = [
            (detector_1_in_L2A_group, 10),
            (detector_2_in_L2A_group, 20),
            (detector_1_in_L2B_group, 20),
            (detector_1_in_L1B14_group, 10),
        ]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(
            detector_to_pha_value
        )
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(
            range=DetectedRange(DetectorRange.R2, DetectorSide.B),
            included_detector_groups=["L2B", "L2A", "L1B14"],
            excluded_detector_groups=[
                "L1A14",
                "L1A0",
                "L3B",
                "L1B0",
                "L4iA",
                "L4iB",
                "L4oB",
            ],
        )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L2A", "L2B", "L1B14"})

        expected_highest_words = [
            detector_to_word_tuples[str(detector_1_in_L2B_group)],
            detector_to_word_tuples[str(detector_2_in_L2A_group)],
            detector_to_word_tuples[str(detector_1_in_L1B14_group)],
        ]

        expected_event_analysis = EventAnalysis(
            range=rule_stub.range,
            l1_detector=detector_1_in_L1B14_group,
            l2_detector=detector_1_in_L2B_group,
            e_delta_word=detector_to_word_tuples[str(detector_1_in_L1B14_group)],
            e_prime_word=detector_to_word_tuples[str(detector_1_in_L2B_group)],
            words_with_highest_energy=expected_highest_words,
        )

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_range_with_range_4(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")
        detector_1_in_group_4 = Detector(layer=3, side="B", segment="a", address=21, group="L3B")
        detector_1_in_group_5 = Detector(layer=2, side="B", segment="a", address=22, group="L2B")

        detector_to_pha_value = [
            (detector_1_in_group_1, 10),
            (detector_2_in_group_1, 20),
            (detector_1_in_group_2, 20),
            (detector_1_in_group_3, 11),
            (detector_1_in_group_4, 12),
            (detector_1_in_group_5, 13),
        ]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(
            detector_to_pha_value
        )
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(
            range=DetectedRange(DetectorRange.R4, DetectorSide.A),
            included_detector_groups=["L1A14", "L2A", "L3A", "L3B", "L2B"],
            excluded_detector_groups=["L1B"],
        )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A", "L3B", "L2B"})

        expected_highest_words = [
            detector_to_word_tuples[str(detector_2_in_group_1)],
            detector_to_word_tuples[str(detector_1_in_group_2)],
            detector_to_word_tuples[str(detector_1_in_group_3)],
            detector_to_word_tuples[str(detector_1_in_group_4)],
            detector_to_word_tuples[str(detector_1_in_group_5)],
        ]
        expected_event_analysis = EventAnalysis(
            range=rule_stub.range,
            l1_detector=detector_2_in_group_1,
            l2_detector=detector_1_in_group_2,
            e_delta_word=detector_to_word_tuples[str(detector_1_in_group_3)],
            e_prime_word=detector_to_word_tuples[str(detector_1_in_group_4)],
            words_with_highest_energy=expected_highest_words,
        )

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_event_uses_detectors_with_highest_adc_values(self):

        detector_l1ab = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_l1ac = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_l2ac = Detector(layer=2, side="A", segment="c", address=15, group="L2A")

        word_l1ab = PHAWord(
            adc_value=1,
            detector=detector_l1ab,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )
        word_l1ac = PHAWord(
            adc_value=2,
            detector=detector_l1ac,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )
        word_l2ac = PHAWord(
            adc_value=3,
            detector=detector_l2ac,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )

        test_cases = [
            (
                "Higher ADC value is used both high gain",
                1,
                False,
                2,
                False,
                detector_l1ac,
                word_l1ac,
            ),
            (
                "Higher ADC value is used both low gain",
                2,
                True,
                1,
                True,
                detector_l1ab,
                word_l1ab,
            ),
            (
                "L1ab Low gain ADC * 20 is greater than L1ac ADC",
                1,
                True,
                19,
                False,
                detector_l1ab,
                word_l1ab,
            ),
            (
                "L1ab Low gain ADC * 20 is less than L1ac ADC",
                1,
                True,
                21,
                False,
                detector_l1ac,
                word_l1ac,
            ),
            (
                "Filters out saturated high gain words",
                2047,
                False,
                1,
                False,
                detector_l1ac,
                word_l1ac,
            ),
        ]

        for (
            name,
            l1ab_adc,
            l1ab_is_low_gain,
            l1ac_adc,
            l1ac_is_low_gain,
            expected_l1_detector,
            expected_l1_word,
        ) in test_cases:
            with self.subTest(name):
                word_l1ab.adc_value = l1ab_adc
                word_l1ab.is_low_gain = l1ab_is_low_gain
                word_l1ac.adc_value = l1ac_adc
                word_l1ac.is_low_gain = l1ac_is_low_gain

                raw_pha_event = create_raw_pha_event(pha_words=[word_l1ab, word_l1ac, word_l2ac])

                rule_stub = Rule(
                    range=DetectedRange(DetectorRange.R2, DetectorSide.A),
                    included_detector_groups=["L1A14", "L2A"],
                    excluded_detector_groups=["L1B"],
                )
                range_lookup_table = Mock()
                range_lookup_table.lookup_range.return_value = rule_stub

                event_analysis = analyze_event(raw_pha_event, range_lookup_table)

                expected_event_analysis = EventAnalysis(
                    range=rule_stub.range,
                    l1_detector=expected_l1_detector,
                    l2_detector=detector_l2ac,
                    e_delta_word=expected_l1_word,
                    e_prime_word=word_l2ac,
                    words_with_highest_energy=[expected_l1_word, word_l2ac],
                )

                self.assertEqual(expected_event_analysis.range, event_analysis.range)
                self.assertEqual(expected_event_analysis.l1_detector, event_analysis.l1_detector)
                self.assertEqual(expected_event_analysis.l2_detector, event_analysis.l2_detector)
                self.assertEqual(expected_event_analysis.e_delta_word, event_analysis.e_delta_word)
                self.assertEqual(expected_event_analysis.e_prime_word, event_analysis.e_prime_word)
                self.assertEqual(
                    expected_event_analysis.words_with_highest_energy,
                    event_analysis.words_with_highest_energy,
                )

    def test_analyze_event_filters_out_events_with_all_saturated_data(self):
        detector_l1ab = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_l1ac = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_l2ac = Detector(layer=2, side="A", segment="c", address=15, group="L2A")

        word_l1ab = PHAWord(
            adc_value=2047,
            detector=detector_l1ab,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )
        word_l1ac = PHAWord(
            adc_value=2047,
            detector=detector_l1ac,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )
        word_l2ac = PHAWord(
            adc_value=1,
            detector=detector_l2ac,
            is_low_gain=False,
            adc_overflow=False,
            is_last_pha=False,
        )

        raw_pha_event = create_raw_pha_event(pha_words=[word_l1ab, word_l1ac, word_l2ac])

        rule_stub = Rule(
            range=DetectedRange(DetectorRange.R2, DetectorSide.A),
            included_detector_groups=["L1A14", "L2A"],
            excluded_detector_groups=["L1B"],
        )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, range_lookup_table)

        self.assertIsNone(event_analysis)

    def test_analyze_event_handles_no_calc_rules(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")

        detector_to_pha_value = [
            (detector_1_in_group_1, 10),
            (detector_2_in_group_1, 20),
            (detector_1_in_group_2, 20),
            (detector_1_in_group_3, 10),
        ]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(
            detector_to_pha_value
        )
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = None

        event_analysis = analyze_event(raw_pha_event, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A"})

        self.assertIsNone(event_analysis)

    @patch(f"{MODULE}.analyze_event")
    @patch(f"{MODULE}.calculate_mev")
    def test_process_event_for_event_with_invalid_detector_sequences(
        self, mock_calculate_mev, mock_analyze_event
    ):
        raw_pha_event = create_raw_pha_event(
            pha_words=[
                PHAWord(
                    detector=Detector.from_address(12),
                    adc_value=10,
                    adc_overflow=False,
                    is_low_gain=True,
                    is_last_pha=False,
                ),
                PHAWord(
                    detector=Detector.from_address(28),
                    adc_value=20,
                    adc_overflow=False,
                    is_low_gain=False,
                    is_last_pha=False,
                ),
            ]
        )

        word1_mev = 101.3
        word2_mev = 66.8

        mock_calculate_mev.side_effect = [word1_mev, word2_mev]
        mock_analyze_event.return_value = None

        event_output = process_pha_event(raw_pha_event, Mock(), Mock(), Mock(), Mock(), Mock())

        expected_event_output = EventOutput(
            original_event=raw_pha_event,
            energies=[word1_mev, word2_mev],
            charge=None,
            total_energy=None,
            detected_range=None,
            e_delta=None,
            e_prime=None,
        )
        self.assertEqual(expected_event_output, event_output)

    @patch(f"{MODULE}.compute_charge")
    @patch(f"{MODULE}.analyze_event", spec_set=True)
    def test_process_pha_event_sums_window_corrected_energies_and_cosine_corrects_only_delta_e_and_e_prime(
        self, mock_analyze_event, mock_compute_charge
    ):
        word_19_adc = 53
        word_8_adc = 92
        word_7_adc = 48

        word_19 = PHAWord(
            detector=Detector.from_address(19),
            adc_value=word_19_adc,
            adc_overflow=False,
            is_low_gain=True,
            is_last_pha=False,
        )
        word_8 = PHAWord(
            detector=Detector.from_address(8),
            adc_value=word_8_adc,
            adc_overflow=False,
            is_low_gain=False,
            is_last_pha=False,
        )
        word_7 = PHAWord(
            detector=Detector.from_address(7),
            adc_value=word_7_adc,
            adc_overflow=False,
            is_low_gain=False,
            is_last_pha=False,
        )

        raw_pha_event = create_raw_pha_event(pha_words=[word_19, word_8, word_7])

        gain_19 = Gain(a=10.76, b=14.3)
        gain_8 = Gain(a=25.89, b=11.2)
        gain_7 = Gain(a=31.3, b=5.9)
        gain_lookup_table = {
            DetectorGain.HIGH: {8: gain_8, 7: gain_7},
            DetectorGain.LOW: {19: gain_19},
        }

        range_2_wincorr = np.full(64, 256)
        range_2_wincorr[19] = 300
        range_2_wincorr[8] = 270
        range_3_and_4_wincorr = np.full(64, 256)
        range_3_and_4_wincorr[19] = 500
        range_3_and_4_wincorr[8] = 400
        window_correction_table = WindowCorrectionLookupTable(
            range_2_wincorr, range_3_and_4_wincorr
        )

        cosine_k = 0.93
        mock_cosine_correction_table = Mock()
        mock_cosine_correction_table.get_cosine_correction.return_value = cosine_k

        mev_19 = word_19_adc * gain_19.a + gain_19.b
        mev_8 = word_8_adc * gain_8.a + gain_8.b
        mev_7 = word_7_adc * gain_7.a + gain_7.b

        cases = [
            (DetectedRange(DetectorRange.R2, DetectorSide.A), range_2_wincorr),
            (DetectedRange(DetectorRange.R3, DetectorSide.A), range_3_and_4_wincorr),
            (DetectedRange(DetectorRange.R4, DetectorSide.B), range_3_and_4_wincorr),
        ]
        for detected_range, wincorr_row in cases:
            with self.subTest(detected_range=detected_range):
                mock_analyze_event.reset_mock()
                mock_compute_charge.reset_mock()
                mock_cosine_correction_table.reset_mock(return_value=True)
                mock_cosine_correction_table.get_cosine_correction.return_value = cosine_k
                mock_analyze_event.return_value = EventAnalysis(
                    range=detected_range,
                    l1_detector=word_19.detector,
                    l2_detector=word_8.detector,
                    e_delta_word=word_19,
                    e_prime_word=word_8,
                    words_with_highest_energy=[word_19, word_8],
                )

                event_output = process_pha_event(
                    raw_pha_event,
                    mock_cosine_correction_table,
                    gain_lookup_table,
                    window_correction_table,
                    sentinel.range_fit_lookup,
                    sentinel.rule_lookup,
                )

                mock_analyze_event.assert_called_once_with(raw_pha_event, sentinel.rule_lookup)
                mock_cosine_correction_table.get_cosine_correction.assert_called_once_with(
                    detected_range, word_19.detector, word_8.detector
                )

                self.assertEqual(raw_pha_event, event_output.original_event)
                self.assertEqual(detected_range, event_output.detected_range)
                self.assertEqual(mock_compute_charge.return_value, event_output.charge)
                np.testing.assert_allclose(event_output.energies, [mev_19, mev_8, mev_7])
                np.testing.assert_allclose(
                    event_output.total_energy,
                    wincorr_row[19] / 256 * mev_19 + wincorr_row[8] / 256 * mev_8,
                )
                np.testing.assert_allclose(event_output.e_delta, cosine_k * mev_19)
                np.testing.assert_allclose(event_output.e_prime, cosine_k * mev_8)
                mock_compute_charge.assert_called_once_with(
                    detected_range,
                    cosine_k * mev_19,
                    cosine_k * mev_8,
                    sentinel.range_fit_lookup,
                )

    def test_compute_charge(self):
        charges = [3, 4, 5, 6]
        delta_e_losses = [1, 2, 20, 200]
        range_2_side_a = DetectedRange(range=DetectorRange.R2, side=DetectorSide.A)
        cases = [
            (range_2_side_a, 15.0, 200, 1, 2),
            (range_2_side_a, 2, 200, 1, 2),
            (range_2_side_a, 2, 200, 0, 1),
            (range_2_side_a, 1.1, 200, 0, 1),
            (range_2_side_a, 0.2, 200, 0, 1),
            (range_2_side_a, 300, 200, 2, 3),
            (range_2_side_a, 430.1, 860.1, 2, 3),
            (DetectedRange(DetectorRange.R2, DetectorSide.B), 0.09, 0.19, 0, 1),
            (DetectedRange(DetectorRange.R3, DetectorSide.A), 860.1, 4300.1, 2, 3),
            (DetectedRange(DetectorRange.R3, DetectorSide.B), 0.19, 0.9, 0, 1),
            (DetectedRange(DetectorRange.R4, DetectorSide.A), 4300.1, 4300.1, 2, 3),
            (DetectedRange(DetectorRange.R4, DetectorSide.B), 0.9, 0.9, 0, 1),
        ]
        for detected_range, delta_e, e_prime, index_1, index_2 in cases:
            with self.subTest(detected_range=detected_range, delta_e=delta_e, e_prime=e_prime):
                mock_double_power_law_fit_params = Mock()
                mock_double_power_law_fit_params.evaluate_e_prime.return_value = (
                    charges,
                    delta_e_losses,
                )

                charge = compute_charge(
                    detected_range, delta_e, e_prime, mock_double_power_law_fit_params
                )

                mock_double_power_law_fit_params.evaluate_e_prime.assert_called_once_with(
                    detected_range, e_prime
                )

                B = math.log(charges[index_2] / charges[index_1]) / math.log(
                    delta_e_losses[index_2] / delta_e_losses[index_1]
                )
                A = charges[index_1] / (delta_e_losses[index_1] ** B)

                self.assertAlmostEqual(charge, A * delta_e**B)

    def test_compute_charge_is_nan_when_delta_e_or_e_prime_is_not_positive(self):
        mock_lookup = Mock()
        mock_lookup.evaluate_e_prime.return_value = (
            np.array([3, 4, 5, 6]),
            np.array([1, 2, 20, 200]),
        )
        detected_range = DetectedRange(DetectorRange.R2, DetectorSide.A)
        cases = [
            (0.0, 50.0),
            (-1.5, 50.0),
            (5.0, 0.0),
            (5.0, -3.0),
        ]
        for delta_e, e_prime in cases:
            with self.subTest(delta_e=delta_e, e_prime=e_prime):
                charge = compute_charge(detected_range, delta_e, e_prime, mock_lookup)
                self.assertTrue(np.isnan(charge))

    def test_compute_charge_is_nan_when_tracks_at_e_prime_are_not_strictly_increasing(self):
        detected_range = DetectedRange(DetectorRange.R3, DetectorSide.A)
        track_values = [
            [1, 20, 2, 200],
            [1, 2, 2, 200],
            [1, np.nan, 20, 200],
        ]
        for values in track_values:
            with self.subTest(track_values=values):
                mock_lookup = Mock()
                mock_lookup.evaluate_e_prime.return_value = (
                    np.array([3, 4, 5, 6]),
                    np.array(values, dtype=float),
                )
                charge = compute_charge(detected_range, 5.0, 50.0, mock_lookup)
                self.assertTrue(np.isnan(charge))

    def _create_detector_from_string(
        self, detector_string: str, detector_group: str | None = None
    ):
        layer = int(detector_string[1])
        group = detector_group if detector_group is not None else "L1A4c"

        return Detector(
            layer=layer,
            side=detector_string[2],
            segment=detector_string[3:],
            address=0,
            group=group,
        )

    def _create_event_from_detectors_and_groups(
        self, detectors: list[str], groups: list[str] | None = None
    ):
        if groups and len(detectors) != len(groups):
            self.assertTrue(False, msg="Detectors and groups must have the same length in setup")

        detector_objects = [
            self._create_detector_from_string(d, g) for d, g in zip(detectors, groups)
        ]
        pha_words = [
            PHAWord(
                adc_value=0,
                adc_overflow=False,
                detector=d,
                is_low_gain=False,
                is_last_pha=False,
            )
            for d in detector_objects
        ]
        raw_pha_event = create_raw_pha_event(pha_words=pha_words)
        return raw_pha_event

    def _create_event_from_detector_to_pha_value_dict(
        self, detector_to_pha_tuples: list[tuple[Detector, int]]
    ):
        pha_words = []
        detector_to_word_tuples = {}
        for detector, value in detector_to_pha_tuples:
            word = PHAWord(
                adc_value=value,
                adc_overflow=False,
                detector=detector,
                is_low_gain=False,
                is_last_pha=False,
            )
            pha_words.append(detector)
            detector_to_word_tuples[str(detector)] = word

        return detector_to_word_tuples
