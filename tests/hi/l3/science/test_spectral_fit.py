import unittest

import numpy as np

from imap_l3_processing.hi.l3.science.spectral_fit import power_law, spectral_fit
from imap_l3_processing.models import InputMetadata


class TestHiProcessor(unittest.TestCase):
    def test_power_law_function(self):
        params = (2, -2)
        x = np.array([1, 2, 3])
        y = np.array([4, 10, 22])
        err = np.array([2, 2, 2])
        keywords = {'xval': x, 'yval': y, 'errval': err}

        expected_residual = np.array([1, 1, 2])
        status, actual_residuals = power_law(params, **keywords)

        np.testing.assert_array_equal(actual_residuals, expected_residual)
        self.assertEqual(status, 0)

    def test_finds_best_fit(self):
        np.random.seed(42)
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))
