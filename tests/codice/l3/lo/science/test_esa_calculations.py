import unittest
from unittest.mock import sentinel

from imap_l3_processing.codice.l3.lo.science.esa_calculations import calculate_partial_densities


class TestEsaCalculations(unittest.TestCase):
    def test_calculate_partial_densities(self):
        result = calculate_partial_densities(sentinel.intensity)
