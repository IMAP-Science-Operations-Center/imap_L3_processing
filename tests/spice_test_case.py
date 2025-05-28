import unittest

import spiceypy

from tests.test_helpers import furnish_local_spice


class SpiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        furnish_local_spice()

    @classmethod
    def tearDownClass(cls):
        spiceypy.kclear()
