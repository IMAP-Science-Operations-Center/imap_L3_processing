import unittest

from imap_processing.spice_wrapper import furnish


class SpiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        furnish()
