import math
import unittest
from collections import defaultdict
from unittest.mock import patch, Mock, sentinel

import numpy as np

from imap_l3_processing.hit.l3.pha.pha_event_reader import PHAWord, Detector
from imap_l3_processing.hit.l3.pha.science.calculate_pha import EventAnalysis, analyze_event, calculate_mev, \
    process_pha_event, EventOutput, compute_charge
from imap_l3_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange, DetectorRange, \
    DetectorSide
from imap_l3_processing.hit.l3.pha.science.gain_lookup_table import DetectorGain, Gain
from imap_l3_processing.hit.l3.pha.science.hit_event_type_lookup import Rule
from imap_l3_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from tests.hit.l3.hit_test_builders import create_raw_pha_event
from tests.test_helpers import get_test_data_path


class TestCalculatePHA(unittest.TestCase):
    def setUp(self) -> None:
        self.gain_lookup = defaultdict(lambda: defaultdict(lambda: Gain(a=-1, b=0)))

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
        low_gain_word = PHAWord(adc_value=low_gain_adc_value, is_low_gain=True, detector=Detector.from_address(10),
                                is_last_pha=False, adc_overflow=False)
        mev = calculate_mev(low_gain_word, gain_lookup_table)
        self.assertEqual(low_gain_a_value * low_gain_adc_value + low_gain_b_value, mev)

        low_gain_word = PHAWord(adc_value=high_gain_adc_value, is_low_gain=False, detector=Detector.from_address(22),
                                is_last_pha=False, adc_overflow=False)
        mev = calculate_mev(low_gain_word, gain_lookup_table)
        self.assertEqual(high_gain_a_value * high_gain_adc_value + high_gain_b_value, mev)

    def test_analyze_range(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")

        detector_to_pha_value = [(detector_1_in_group_1, 10),
                                 (detector_2_in_group_1, 20),
                                 (detector_1_in_group_2, 20),
                                 (detector_1_in_group_3, 10)]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(detector_to_pha_value)
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(range=DetectedRange(DetectorRange.R2, DetectorSide.A),
                         included_detector_groups=["L1A14", "L2A"],
                         excluded_detector_groups=["L2B"]
                         )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, self.gain_lookup, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A"})

        expected_highest_words = [detector_to_word_tuples[str(detector_1_in_group_1)],
                                  detector_to_word_tuples[str(detector_1_in_group_2)]]
        expected_event_analysis = EventAnalysis(range=rule_stub.range,
                                                l1_detector=detector_1_in_group_1,
                                                l2_detector=detector_1_in_group_2,
                                                e_delta_word=detector_to_word_tuples[str(detector_1_in_group_1)],
                                                e_prime_word=detector_to_word_tuples[str(detector_1_in_group_2)],
                                                words_with_highest_energy=expected_highest_words)

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_range_mixed_sides(self):
        detector_2_in_L2A_group = Detector(layer=2, side="A", segment="c", address=11, group="L2A")
        detector_1_in_L2A_group = Detector(layer=2, side="A", segment="b", address=10, group="L2A")
        detector_1_in_L2B_group = Detector(layer=2, side="B", segment="c", address=15, group="L2B")
        detector_1_in_L1B14_group = Detector(layer=1, side="B", segment="a", address=20, group="L1B14")

        detector_to_pha_value = [(detector_1_in_L2A_group, 10),
                                 (detector_2_in_L2A_group, 20),
                                 (detector_1_in_L2B_group, 20),
                                 (detector_1_in_L1B14_group, 10)]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(detector_to_pha_value)
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(range=DetectedRange(DetectorRange.R2, DetectorSide.B),
                         included_detector_groups=["L2B", "L2A", "L1B14"],
                         excluded_detector_groups=["L1A14", "L1A0", "L3B", "L1B0", "L4iA", "L4iB", "L4oB"]
                         )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, self.gain_lookup, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L2A", "L2B", "L1B14"})

        expected_highest_words = [
            detector_to_word_tuples[str(detector_1_in_L2B_group)],
            detector_to_word_tuples[str(detector_1_in_L2A_group)],
            detector_to_word_tuples[str(detector_1_in_L1B14_group)]
            ,

        ]

        expected_event_analysis = EventAnalysis(range=rule_stub.range,
                                                l1_detector=detector_1_in_L1B14_group,
                                                l2_detector=detector_1_in_L2B_group,
                                                e_delta_word=detector_to_word_tuples[str(detector_1_in_L1B14_group)],
                                                e_prime_word=detector_to_word_tuples[str(detector_1_in_L2B_group)],
                                                words_with_highest_energy=expected_highest_words)

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_range_with_range_4(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")
        detector_1_in_group_4 = Detector(layer=3, side="B", segment="a", address=21, group="L3B")
        detector_1_in_group_5 = Detector(layer=2, side="B", segment="a", address=22, group="L2B")

        detector_to_pha_value = [(detector_1_in_group_1, 10),
                                 (detector_2_in_group_1, 20),
                                 (detector_1_in_group_2, 20),
                                 (detector_1_in_group_3, 11),
                                 (detector_1_in_group_4, 12),
                                 (detector_1_in_group_5, 13),
                                 ]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(detector_to_pha_value)
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        rule_stub = Rule(range=DetectedRange(DetectorRange.R4, DetectorSide.A),
                         included_detector_groups=["L1A14", "L2A", "L3A", "L3B", "L2B"],
                         excluded_detector_groups=["L1B"]
                         )
        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = rule_stub

        event_analysis = analyze_event(raw_pha_event, self.gain_lookup, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A", "L3B", "L2B"})

        expected_highest_words = [detector_to_word_tuples[str(detector_1_in_group_1)],
                                  detector_to_word_tuples[str(detector_1_in_group_2)],
                                  detector_to_word_tuples[str(detector_1_in_group_3)],
                                  detector_to_word_tuples[str(detector_1_in_group_4)],
                                  detector_to_word_tuples[str(detector_1_in_group_5)],
                                  ]
        expected_event_analysis = EventAnalysis(range=rule_stub.range,
                                                l1_detector=detector_1_in_group_1,
                                                l2_detector=detector_1_in_group_2,
                                                e_delta_word=detector_to_word_tuples[str(detector_1_in_group_3)],
                                                e_prime_word=detector_to_word_tuples[str(detector_1_in_group_4)],
                                                words_with_highest_energy=expected_highest_words)

        self.assertEqual(expected_event_analysis, event_analysis)

    def test_analyze_event_handles_no_calc_rules(self):
        detector_1_in_group_1 = Detector(layer=1, side="A", segment="b", address=10, group="L1A14")
        detector_2_in_group_1 = Detector(layer=1, side="A", segment="c", address=11, group="L1A14")
        detector_1_in_group_2 = Detector(layer=2, side="A", segment="c", address=15, group="L2A")
        detector_1_in_group_3 = Detector(layer=3, side="A", segment="a", address=20, group="L3A")

        detector_to_pha_value = [(detector_1_in_group_1, 10),
                                 (detector_2_in_group_1, 20),
                                 (detector_1_in_group_2, 20),
                                 (detector_1_in_group_3, 10)]

        detector_to_word_tuples = self._create_event_from_detector_to_pha_value_dict(detector_to_pha_value)
        words = [word for _, word in detector_to_word_tuples.items()]
        raw_pha_event = create_raw_pha_event(pha_words=words)

        range_lookup_table = Mock()
        range_lookup_table.lookup_range.return_value = None

        event_analysis = analyze_event(raw_pha_event, self.gain_lookup, range_lookup_table)

        range_lookup_table.lookup_range.assert_called_with({"L1A14", "L2A", "L3A"})

        self.assertIsNone(event_analysis)

    @patch("imap_l3_processing.hit.l3.pha.science.calculate_pha.analyze_event")
    @patch("imap_l3_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
    def test_process_event_for_event_with_invalid_detector_sequences(self, mock_calculate_mev, mock_analyze_event):
        raw_pha_event = create_raw_pha_event(pha_words=[
            PHAWord(detector=Detector.from_address(12), adc_value=10, adc_overflow=False,
                    is_low_gain=True, is_last_pha=False),
            PHAWord(detector=Detector.from_address(28), adc_value=20,
                    adc_overflow=False,
                    is_low_gain=False, is_last_pha=False)])

        word1_mev = 101.3
        word2_mev = 66.8

        mock_calculate_mev.side_effect = [word1_mev, word2_mev]
        mock_analyze_event.return_value = None

        event_output = process_pha_event(raw_pha_event, Mock(), Mock(), Mock(), Mock())

        expected_event_output = EventOutput(original_event=raw_pha_event, energies=[word1_mev, word2_mev], charge=None,
                                            total_energy=None, detected_range=None, e_delta=None, e_prime=None, )
        self.assertEqual(expected_event_output, event_output)

    @patch("imap_l3_processing.hit.l3.pha.science.calculate_pha.compute_charge")
    @patch("imap_l3_processing.hit.l3.pha.science.calculate_pha.analyze_event", spec_set=True)
    def test_process_pha_event(self, mock_analyze_event, mock_compute_charge):
        pha_word1_adc_value = 53
        pha_word2_adc_value = 92
        pha_word3_adc_value = 48

        l1_word = PHAWord(detector=Detector.from_address(12), adc_value=pha_word1_adc_value, adc_overflow=False,
                          is_low_gain=True, is_last_pha=False)
        l2_high_energy_word = PHAWord(detector=Detector.from_address(28), adc_value=pha_word2_adc_value,
                                      adc_overflow=False,
                                      is_low_gain=False, is_last_pha=False)
        l2_low_energy_word = PHAWord(detector=Detector.from_address(29), adc_value=pha_word3_adc_value,
                                     adc_overflow=False,
                                     is_low_gain=False, is_last_pha=False)

        raw_pha_event = create_raw_pha_event(pha_words=[l1_word, l2_high_energy_word, l2_low_energy_word])

        word1_gain = Gain(a=10.76, b=14.3)
        word2_gain = Gain(a=25.89, b=11.2)
        word3_gain = Gain(a=31.3, b=5.9)

        mock_analyze_event.return_value = EventAnalysis(range=sentinel.detected_range, l1_detector=sentinel.l1_detector,
                                                        l2_detector=sentinel.l2_detector,
                                                        e_delta_word=l1_word,
                                                        e_prime_word=l2_high_energy_word,
                                                        words_with_highest_energy=[l1_word, l2_high_energy_word])

        gain_lookup_table = {
            DetectorGain.HIGH: {28: word2_gain, 29: word3_gain},
            DetectorGain.LOW: {12: word1_gain}
        }

        cosine_correction = 12.2

        mock_cosine_correction_table = Mock()
        mock_cosine_correction_table.get_cosine_correction.return_value = cosine_correction

        event_output = process_pha_event(raw_pha_event, mock_cosine_correction_table, gain_lookup_table,
                                         sentinel.range_fit_lookup, sentinel.rule_lookup)

        mock_analyze_event.assert_called_once_with(raw_pha_event, gain_lookup_table, sentinel.rule_lookup)

        mock_cosine_correction_table.get_cosine_correction.assert_called_once_with(sentinel.detected_range,
                                                                                   sentinel.l1_detector,
                                                                                   sentinel.l2_detector)

        self.assertIsInstance(event_output, EventOutput)

        expected_l1_energy = cosine_correction * (word1_gain.a * pha_word1_adc_value + word1_gain.b)
        expected_l2_higher_energy = cosine_correction * (word2_gain.a * pha_word2_adc_value + word2_gain.b)
        expected_energy3 = cosine_correction * (word3_gain.a * pha_word3_adc_value + word3_gain.b)
        self.assertEqual([expected_l1_energy, expected_l2_higher_energy, expected_energy3], event_output.energies)

        self.assertEqual(expected_l1_energy + expected_l2_higher_energy, event_output.total_energy)
        self.assertEqual(raw_pha_event, event_output.original_event)

        mock_compute_charge.assert_called_once_with(sentinel.detected_range, expected_l1_energy,
                                                    expected_l2_higher_energy, sentinel.range_fit_lookup)
        self.assertEqual(mock_compute_charge.return_value, event_output.charge)
        self.assertEqual(sentinel.detected_range, event_output.detected_range)
        self.assertEqual(expected_l1_energy, event_output.e_delta)
        self.assertEqual(expected_l2_higher_energy, event_output.e_prime)

    def test_compute_charge(self):
        charges = [3, 4, 5, 6]
        delta_e_losses = [1, 2, 20, 200]
        cases = [
            (15.0, 1, 2),
            (2, 1, 2),
            (2, 0, 1),
            (1.1, 0, 1),
            (0.2, 0, 1),
            (300, 2, 3),
        ]
        for delta_e, index_1, index_2 in cases:
            with self.subTest(delta_e):
                mock_double_power_law_fit_params = Mock()
                mock_double_power_law_fit_params.evaluate_e_prime.return_value = (charges, delta_e_losses)

                detected_range = DetectedRange(range=DetectorRange.R2, side=DetectorSide.A)
                e_prime = 200
                charge = compute_charge(detected_range, delta_e, e_prime,
                                        mock_double_power_law_fit_params)

                mock_double_power_law_fit_params.evaluate_e_prime.assert_called_once_with(detected_range,
                                                                                          e_prime)

                B = math.log(charges[index_2] / charges[index_1]) / math.log(
                    delta_e_losses[index_2] / delta_e_losses[index_1])
                A = charges[index_1] / (delta_e_losses[index_1] ** B)

                self.assertAlmostEqual(charge, A * delta_e ** B)

    def test_compute_charge_returns_nan_if_e_prime_or_delta_e_are_outside_of_range(self):
        detected_range_1 = DetectedRange(range=DetectorRange.R2, side=DetectorSide.A)
        detected_range_2 = DetectedRange(range=DetectorRange.R3, side=DetectorSide.B)
        detected_range_3 = DetectedRange(range=DetectorRange.R4, side=DetectorSide.A)
        range_fit_lookup_table = RangeFitLookup.from_files(
            get_test_data_path("hit/pha_events/imap_hit_range-2A-charge-fit-lookup_20250319_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-3A-charge-fit-lookup_20250319_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-4A-charge-fit-lookup_20250319_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-2B-charge-fit-lookup_20250319_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-3B-charge-fit-lookup_20250319_v000.csv"),
            get_test_data_path("hit/pha_events/imap_hit_range-4B-charge-fit-lookup_20250319_v000.csv"),
        )
        test_cases = [
            ("detector 1 - delta_e over max", detected_range_1, 430.1, 860, True),
            ("detector 1 - delta_e at max", detected_range_1, 430, 860, False),
            ("detector 1 - delta_e at min", detected_range_1, 0.1, 860, False),
            ("detector 1 - delta_e under min", detected_range_1, 0.09, 860, True),
            ("detector 1 - e_prime over max", detected_range_1, 430, 860.1, True),
            ("detector 1 - e_prime at max", detected_range_1, 430, 860, False),
            ("detector 1 - e_prime at min", detected_range_1, 430, 0.2, False),
            ("detector 1 - e_prime under min", detected_range_1, 430, 0.19, True),
            ("detector 2 - delta_e over max", detected_range_2, 860.1, 430, True),
            ("detector 2 - delta_e at max", detected_range_2, 860, 430, False),
            ("detector 2 - delta_e at min", detected_range_2, 0.2, 860, False),
            ("detector 2 - delta_e under min", detected_range_2, 0.19, 860, True),
            ("detector 2 - e_prime over max", detected_range_2, 430, 4300.1, True),
            ("detector 2 - e_prime at max", detected_range_2, 430, 4300, False),
            ("detector 2 - e_prime at min", detected_range_2, 1, 1.0, False),
            ("detector 2 - e_prime under min", detected_range_2, 430, 0.9, True),
            ("detector 3 - delta_e over max", detected_range_3, 4300.1, 860, True),
            ("detector 3 - delta_e at max", detected_range_3, 4300, 860, False),
            ("detector 3 - delta_e at min", detected_range_3, 1.0, 860, False),
            ("detector 3 - delta_e under min", detected_range_3, 0.09, 860, True),
            ("detector 3 - e_prime over max", detected_range_3, 430, 4300.1, True),
            ("detector 3 - e_prime at max", detected_range_3, 430, 4300, False),
            ("detector 3 - e_prime at min", detected_range_3, 430, 1.0, False),
            ("detector 3 - e_prime under min", detected_range_3, 430, 0.9, True),
        ]
        for test_name, detected_range, delta_e, e_prime, expected_nan in test_cases:
            with self.subTest(test_name):
                result = compute_charge(detected_range, delta_e=delta_e, e_prime=e_prime,
                                        double_power_law_lookup=range_fit_lookup_table)
                self.assertEqual(expected_nan, np.isnan(result))

    def _create_detector_from_string(self, detector_string: str, detector_group: str = None):
        layer = int(detector_string[1])
        group = detector_group if detector_group is not None else "L1A4c"

        return Detector(layer=layer,
                        side=detector_string[2],
                        segment=detector_string[3:],
                        address=0,
                        group=group)

    def _create_event_from_detectors_and_groups(self, detectors: list[str], groups: list[str] = None):
        if groups and len(detectors) != len(groups):
            self.assertTrue(False, msg="Detectors and groups must have the same length in setup")

        detector_objects = [self._create_detector_from_string(d, g) for d, g in zip(detectors, groups)]
        pha_words = [PHAWord(adc_value=0, adc_overflow=False, detector=d, is_low_gain=False, is_last_pha=False) for d in
                     detector_objects]
        raw_pha_event = create_raw_pha_event(pha_words=pha_words)
        return raw_pha_event

    def _create_event_from_detector_to_pha_value_dict(self, detector_to_pha_tuples: list[tuple[Detector, int]]):
        pha_words = []
        detector_to_word_tuples = {}
        for detector, value in detector_to_pha_tuples:
            word = PHAWord(adc_value=value, adc_overflow=False, detector=detector, is_low_gain=False, is_last_pha=False)
            pha_words.append(detector)
            detector_to_word_tuples[str(detector)] = word

        return detector_to_word_tuples
