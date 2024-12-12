import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imap_processing.glows.l3a.science.bad_angle_flag_configuration import BadAngleFlagConfiguration
from tests.test_helpers import get_test_data_path


class TestBadAngleFlagConfiguration(unittest.TestCase):
    def test_masks_data_when_flags_match_configuration(self):
        close_to_uv_source = [False, False, True, True, False]
        excluded_region = [True, False, True, False, False]
        excluded_by_instr_team = [False, False, False, True, False]
        suspected_transient = [False, True, True, False, False]
        test_data = np.array([
            close_to_uv_source,
            excluded_region,
            excluded_by_instr_team,
            suspected_transient,
        ])

        test_cases = [([True, True, False, True], [True, True, True, True, False]),
                      ([True, False, False, False], close_to_uv_source),
                      ([False, True, False, False], excluded_region),
                      ([False, False, True, False], excluded_by_instr_team),
                      ([False, False, False, True], suspected_transient),
                      ([False, True, False, True], [a or b for a, b in zip(excluded_region, suspected_transient)]),
                      ([False, False, False, False], [False, False, False, False, False]),
                      ]

        for flag_configuration, expected_mask in test_cases:
            with self.subTest(f"Flag configuration: {flag_configuration}"):
                configuration = BadAngleFlagConfiguration(*flag_configuration)
                masked = configuration.evaluate_flags(test_data)
                np.testing.assert_equal(masked, expected_mask, strict=True)

    def test_load_from_file(self):
        path = get_test_data_path(
            "glows/imap_glows_l2_histogram-bad-angle-flags-configuration-json-not-cdf_20250701_v001.cdf")
        expected = BadAngleFlagConfiguration(
            mask_close_to_uv_source=True,
            mask_inside_excluded_region=True,
            mask_excluded_by_instr_team=True,
            mask_suspected_transient=True,
        )
        self.assertEqual(expected, BadAngleFlagConfiguration.from_file(path))

    def test_load_from_file_v2(self):
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = Path(tempdir) / "some_flag_configuration.json"
            expected = BadAngleFlagConfiguration(
                mask_close_to_uv_source=True,
                mask_inside_excluded_region=False,
                mask_excluded_by_instr_team=True,
                mask_suspected_transient=False,
            )
            with open(temp_file, 'w') as f:
                json.dump(dataclasses.asdict(expected), f)
            self.assertEqual(expected, BadAngleFlagConfiguration.from_file(temp_file))
