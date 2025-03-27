import unittest
from unittest.mock import Mock

from imap_l3_processing.glows.glows_initializer import GlowsInitializer
from imap_l3_processing.glows.l3b.glows_l3b_dependencies import GlowsInitializerDependencies


class TestGlowsInitializer(unittest.TestCase):

    def test_returns_false_if_any_external_dependency_file_is_not_found(self):
        test_cases = [
            (None, Mock(), Mock(), False),
            (Mock(), None, Mock(), False),
            (Mock(), Mock(), None, False),
            (None, None, None, False),
            (Mock(), Mock(), Mock(), True)
        ]

        for f107_path, lyman_alpha_path, omni_data_path, should_process in test_cases:
            with self.subTest(
                    f"f107 {f107_path}, lyman_alpha_path {lyman_alpha_path}, omni_data_path {omni_data_path}"):
                ancillary_files = {
                    "f107_index": f107_path,
                    "lyman_alpha_composite_index": lyman_alpha_path,
                    "omni2_data": omni_data_path
                }

                glows_dependency = GlowsInitializerDependencies(Mock(), Mock(), ancillary_files)

                initializer = GlowsInitializer()
                self.assertEqual(should_process, initializer.should_process(glows_dependency))
