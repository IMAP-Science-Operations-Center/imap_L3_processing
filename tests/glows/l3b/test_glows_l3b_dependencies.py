import unittest

from imap_l3_processing.glows.l3b.glows_l3b_dependencies import GlowsL3BDependencies


class TestGlowsL3BDependencies(unittest.TestCase):
    def test_fetch_dependencies(self):
        actual_dependencies = GlowsL3BDependencies.fetch_dependencies()
        self.assertIsInstance(actual_dependencies, GlowsL3BDependencies)
