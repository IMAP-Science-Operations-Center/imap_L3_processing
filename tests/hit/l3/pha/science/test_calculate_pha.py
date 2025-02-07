import math
import unittest
from collections import defaultdict
from unittest.mock import patch, call, Mock, sentinel

from imap_processing.hit.l3.pha.pha_event_reader import PHAWord, Detector
from imap_processing.hit.l3.pha.science.calculate_pha import EventAnalysis, analyze_event, calculate_mev, \
    process_pha_event, EventOutput, compute_charge
from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import DetectedRange
from imap_processing.hit.l3.pha.science.gain_lookup_table import DetectorGain, Gain
from tests.hit.l3.hit_test_builders import create_raw_pha_event


class TestCalculatePHA(unittest.TestCase):
    def setUp(self) -> None:
        self.gain_lookup = defaultdict(lambda: defaultdict(lambda: Gain(a=1, b=0)))

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

    def test_analyze_range2_simple_case(self):
        event = self._create_event_from_detectors(["L1A1c", "L2A2"])

        event_analysis = analyze_event(event, self.gain_lookup)

        expected_event_analysis = EventAnalysis(range=DetectedRange.R2,
                                                l1_detector=event.pha_words[0].detector,
                                                l2_detector=event.pha_words[1].detector,
                                                e_delta_word=event.pha_words[0],
                                                e_prime_word=event.pha_words[1],
                                                words_with_highest_energy=event.pha_words)
        self.assertEqual(expected_event_analysis, event_analysis)

        raw_pha_event_flipped = self._create_event_from_detectors(["L2A2", "L1A1c"])
        self.assertEqual(expected_event_analysis, analyze_event(raw_pha_event_flipped, self.gain_lookup))

    def test_analyzes_range4_event(self):
        event = self._create_event_from_detectors(["L1A3b", "L2A4", "L3Ao", "L3Bo"])
        l1_word, l2_word, l3a_word, l3b_word = event.pha_words
        expected_event_analysis = EventAnalysis(range=DetectedRange.R4,
                                                l1_detector=l1_word.detector,
                                                l2_detector=l2_word.detector,
                                                e_delta_word=l3a_word,
                                                e_prime_word=l3b_word,
                                                words_with_highest_energy=event.pha_words,
                                                )
        self.assertEqual(expected_event_analysis, analyze_event(event, self.gain_lookup))

        flipped_l3_detector_order = self._create_event_from_detectors(["L1A3b", "L2A4", "L3Bo", "L3Ao"])
        self.assertEqual(expected_event_analysis, analyze_event(flipped_l3_detector_order, self.gain_lookup))

    def test_analyzes_range4_event_from_b_side(self):
        event = self._create_event_from_detectors(["L1B3b", "L2B4", "L3Bo", "L3Ao"])
        l1_word, l2_word, l3b_word, l3a_word = event.pha_words
        expected_event_analysis = EventAnalysis(range=DetectedRange.R4,
                                                l1_detector=l1_word.detector,
                                                l2_detector=l2_word.detector,
                                                e_delta_word=l3b_word,
                                                e_prime_word=l3a_word,
                                                words_with_highest_energy=event.pha_words
                                                )
        self.assertEqual(expected_event_analysis, analyze_event(event, self.gain_lookup))

        flipped_l3_detector_order = self._create_event_from_detectors(["L1B3b", "L2B4", "L3Ao", "L3Bo"])
        self.assertEqual(expected_event_analysis, analyze_event(flipped_l3_detector_order, self.gain_lookup))

    def test_analyzes_range3_event(self):
        event = self._create_event_from_detectors(["L1A3b", "L2A4", "L3Ao"])
        l1_word, l2_word, l3a_word = event.pha_words
        expected_event_analysis = EventAnalysis(range=DetectedRange.R3,
                                                l1_detector=l1_word.detector,
                                                l2_detector=l2_word.detector,
                                                e_delta_word=l2_word,
                                                e_prime_word=l3a_word,
                                                words_with_highest_energy=event.pha_words
                                                )
        self.assertEqual(expected_event_analysis, analyze_event(event, self.gain_lookup))

        wrong_side_l3 = self._create_event_from_detectors(["L1A3b", "L2A4", "L3Bo"])
        self.assertIsNone(analyze_event(wrong_side_l3, self.gain_lookup))

    def test_analyzes_event_returns_none_when_l1_and_l2_sides_mismatch(self):
        test_cases = [
            ("range 2", ["L1B3b", "L2A4"]),
            ("range 3", ["L1B3b", "L2A4", "L3Ao"]),
            ("range 4", ["L1B3b", "L2A4", "L3Ao", "L3Bo"]),
            ("no l1", ["L2A1"]),
            ("both sides l1", ["L1A2c", "L1B2c", "L2A1"]),
            ("no l2", ["L1A1a"]),
            ("both sides l2", ["L1A2c", "L2B2", "L2A1"]),
            ("no l1 or l2", []),
            ("both sides l1 and l2", ["L1A2c", "L1B2c", "L2B2", "L2A1"]),
        ]

        for name, detectors in test_cases:
            with self.subTest(name):
                event = self._create_event_from_detectors(detectors)
                self.assertIsNone(analyze_event(event, self.gain_lookup))

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
    def test_analyzes_event_range_2_uses_the_largest_energy_detector_for_each_layer(self, mock_calculate_mev):
        detectors = ["L1B3b", "L1B3c", "L2B4", "L2B2"]

        def calculate_mev_side_effect(word: PHAWord, _):
            detector_to_energy = {"L1B3b": 100, "L1B3c": 48, "L2B4": 95, "L2B2": 60}
            return detector_to_energy[str(word.detector)]

        mock_calculate_mev.side_effect = calculate_mev_side_effect

        event = self._create_event_from_detectors(detectors)
        l1_higher_energy_word, _, l2_word, _ = event.pha_words

        event_analysis = analyze_event(event, self.gain_lookup)
        mock_calculate_mev.assert_has_calls([call(word, self.gain_lookup) for word in event.pha_words])

        self.assertEqual(DetectedRange.R2, event_analysis.range)
        self.assertEqual(l1_higher_energy_word.detector, event_analysis.l1_detector)
        self.assertEqual(l2_word.detector, event_analysis.l2_detector)
        self.assertEqual(l1_higher_energy_word, event_analysis.e_delta_word)
        self.assertEqual(l2_word, event_analysis.e_prime_word)
        self.assertEqual([l1_higher_energy_word, l2_word], event_analysis.words_with_highest_energy)

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
    def test_analyzes_event_range_3_uses_the_largest_energy_detector_for_each_layer(self, mock_calculate_mev):
        detectors = ["L1A3b", "L1A3c", "L2A4", "L2A2", "L3Ao", "L3Ai"]

        def calculate_mev_side_effect(word: PHAWord, _):
            detector_to_energy = {"L1A3b": 45, "L1A3c": 102, "L2A4": 93, "L2A2": 58, "L3Ao": 50, "L3Ai": 90}
            return detector_to_energy[str(word.detector)]

        mock_calculate_mev.side_effect = calculate_mev_side_effect

        event = self._create_event_from_detectors(detectors)
        _, l1_higher_energy_word, l2_higher_energy_word, _, _, l3_high_energy_word = event.pha_words

        event_analysis = analyze_event(event, self.gain_lookup)
        mock_calculate_mev.assert_has_calls([call(word, self.gain_lookup) for word in event.pha_words])

        self.assertEqual(DetectedRange.R3, event_analysis.range)
        self.assertEqual(l1_higher_energy_word.detector, event_analysis.l1_detector)
        self.assertEqual(l2_higher_energy_word.detector, event_analysis.l2_detector)
        self.assertEqual(l2_higher_energy_word, event_analysis.e_delta_word)
        self.assertEqual(l3_high_energy_word, event_analysis.e_prime_word)
        self.assertEqual([l1_higher_energy_word, l2_higher_energy_word, l3_high_energy_word],
                         event_analysis.words_with_highest_energy)

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
    def test_analyzes_event_range_4_uses_the_largest_energy_detector_for_each_layer(self, mock_calculate_mev):
        detectors = ["L1A3b", "L1A3c", "L2A4", "L2A2", "L3Ao", "L3Ai", "L3Bo", "L3Bi"]

        def calculate_mev_side_effect(word: PHAWord, _):
            detector_to_energy = {"L1A3b": 45, "L1A3c": 102,
                                  "L2A4": 93, "L2A2": 58,
                                  "L3Ao": 30, "L3Ai": 9,
                                  "L3Bo": 128, "L3Bi": 0.001}
            return detector_to_energy[str(word.detector)]

        mock_calculate_mev.side_effect = calculate_mev_side_effect

        event = self._create_event_from_detectors(detectors)
        _, l1_higher_energy_word, l2_higher_energy_word, _, l3a_high_energy_word, _, l3b_high_energy_word, _ = event.pha_words

        event_analysis = analyze_event(event, self.gain_lookup)
        mock_calculate_mev.assert_has_calls([call(word, self.gain_lookup) for word in event.pha_words])

        self.assertEqual(DetectedRange.R4, event_analysis.range)
        self.assertEqual(l1_higher_energy_word.detector, event_analysis.l1_detector)
        self.assertEqual(l2_higher_energy_word.detector, event_analysis.l2_detector)
        self.assertEqual(l3a_high_energy_word, event_analysis.e_delta_word)
        self.assertEqual(l3b_high_energy_word, event_analysis.e_prime_word)
        self.assertEqual([l1_higher_energy_word, l2_higher_energy_word, l3a_high_energy_word, l3b_high_energy_word],
                         event_analysis.words_with_highest_energy)

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
    def test_analyzes_event_range_4_that_passes_through_both_l3_layers(self, mock_calculate_mev):
        detectors = ["L1A3b", "L2A4", "L3Ao", "L3Bo", "L2B1", "L2B2"]

        def calculate_mev_side_effect(word: PHAWord, _):
            detector_to_energy = {"L2B1": 128, "L2B2": 0.001}
            return detector_to_energy.get(str(word.detector)) or word.adc_value

        mock_calculate_mev.side_effect = calculate_mev_side_effect

        event = self._create_event_from_detectors(detectors)
        l1_higher_energy_word, l2_higher_energy_word, l3a_high_energy_word, l3b_high_energy_word, l2b_high_energy_word, _ = event.pha_words

        event_analysis = analyze_event(event, self.gain_lookup)

        self.assertEqual(DetectedRange.R4, event_analysis.range)
        self.assertEqual(l1_higher_energy_word.detector, event_analysis.l1_detector)
        self.assertEqual(l2_higher_energy_word.detector, event_analysis.l2_detector)
        self.assertEqual(l3a_high_energy_word, event_analysis.e_delta_word)
        self.assertEqual(l3b_high_energy_word, event_analysis.e_prime_word)
        self.assertEqual(
            [l1_higher_energy_word, l2_higher_energy_word, l3a_high_energy_word, l3b_high_energy_word,
             l2b_high_energy_word],
            event_analysis.words_with_highest_energy)

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.analyze_event")
    @patch("imap_processing.hit.l3.pha.science.calculate_pha.calculate_mev")
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

        event_output = process_pha_event(raw_pha_event, Mock(), Mock())

        expected_event_output = EventOutput(original_event=raw_pha_event, energies=[word1_mev, word2_mev], charge=None,
                                            total_energy=None)
        self.assertEqual(expected_event_output, event_output)

    @patch("imap_processing.hit.l3.pha.science.calculate_pha.analyze_event")
    def test_process_pha_event(self, mock_analyze_event):
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
                                                        e_delta_word=sentinel.e_delta_word,
                                                        e_prime_word=sentinel.e_prime_word,
                                                        words_with_highest_energy=[l1_word, l2_high_energy_word])

        gain_lookup_table = {
            DetectorGain.HIGH: {28: word2_gain, 29: word3_gain},
            DetectorGain.LOW: {12: word1_gain}
        }

        cosine_correction = 12.2

        mock_cosine_correction_table = Mock()
        mock_cosine_correction_table.get_cosine_correction.return_value = cosine_correction

        event_output = process_pha_event(raw_pha_event, mock_cosine_correction_table, gain_lookup_table)

        mock_analyze_event.assert_called_once_with(raw_pha_event, gain_lookup_table)

        mock_cosine_correction_table.get_cosine_correction.assert_called_once_with(sentinel.detected_range,
                                                                                   sentinel.l1_detector,
                                                                                   sentinel.l2_detector)

        self.assertIsInstance(event_output, EventOutput)

        expected_energy1 = cosine_correction * (word1_gain.a * pha_word1_adc_value + word1_gain.b)
        expected_energy2 = cosine_correction * (word2_gain.a * pha_word2_adc_value + word2_gain.b)
        expected_energy3 = cosine_correction * (word3_gain.a * pha_word3_adc_value + word3_gain.b)
        self.assertEqual([expected_energy1, expected_energy2, expected_energy3], event_output.energies)

        self.assertEqual(expected_energy1 + expected_energy2, event_output.total_energy)
        self.assertEqual(raw_pha_event, event_output.original_event)

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

                charge = compute_charge(sentinel.detected_range, delta_e, sentinel.e_prime,
                                        mock_double_power_law_fit_params)

                mock_double_power_law_fit_params.evaluate_e_prime.assert_called_once_with(sentinel.detected_range,
                                                                                          sentinel.e_prime)

                B = math.log(charges[index_2] / charges[index_1]) / math.log(
                    delta_e_losses[index_2] / delta_e_losses[index_1])
                A = charges[index_1] / (delta_e_losses[index_1] ** B)

                self.assertAlmostEqual(charge, A * delta_e ** B)

    def _create_detector_from_string(self, detector_string: str):
        layer = int(detector_string[1])
        return Detector(layer=layer,
                        side=detector_string[2],
                        segment=detector_string[3:],
                        address=0)

    def _create_event_from_detectors(self, detectors: list[str]):
        detector_objects = [self._create_detector_from_string(d) for d in detectors]
        pha_words = [PHAWord(adc_value=0, adc_overflow=False, detector=d, is_low_gain=False, is_last_pha=False) for d in
                     detector_objects]
        raw_pha_event = create_raw_pha_event(pha_words=pha_words)
        return raw_pha_event
