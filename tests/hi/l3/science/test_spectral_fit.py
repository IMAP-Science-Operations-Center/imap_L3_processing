import unittest

import numpy as np

from imap_l3_processing.hi.l3.science.spectral_fit import power_law, spectral_fit


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
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))
        np.testing.assert_array_almost_equal(result_error, np.array([0.04120789]).reshape(1, 1, 1))

    def test_finds_best_fit_with_nan_in_flux(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        flux_data[len(flux_data) // 2] = np.nan
        flux_data[0] = np.nan
        flux_data[-1] = np.nan

        errors = 0.2 * np.abs(flux_data)

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))

    def test_finds_best_fit_with_nan_in_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)
        errors[len(errors) // 2] = np.nan
        errors[0] = np.nan
        errors[-1] = np.nan

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))

    def test_finds_best_fit_with_zero_in_flux_and_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, -1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux_data[0:3] = 0
        errors[0:3] = 0

        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, 1))

    def test_finds_best_fit_with_ibex_data(self):
        energies = np.array([0.71, 1.11, 1.74, 2.73, 4.29])
        flux_data = np.array([[[[46.710853, 34.896639, 18.899919, 14.709325, 5.815468],
                                [61.45169, 11.987692, 17.140896, 12.165362, 5.071331]],
                               [[60.682266, 29.323209, 16.12169, 13.114869, 6.060111],
                                [60.523616, 19.239248, 14.099767, 13.726924, 6.499908]]]])
        errors = np.array([[[[4.17239236e+02, 6.38942760e+01, 1.58932960e+01, 2.01988700e+00,
                              3.00438000e-01],
                             [5.87577556e+02, 3.62103490e+01, 1.41284380e+01, 2.75456600e+00,
                              3.76740000e-01]],
                            [[4.59947913e+02, 2.87564900e+01, 1.34992710e+01, 1.83946100e+00,
                              2.15943000e-01],
                             [7.02472212e+02, 5.06517210e+01, 9.66067900e+00, 2.25438200e+00,
                              3.76271000e-01]]]])

        num_lats = 2
        num_lons = 2
        num_epochs = 1

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux_data, errors, energies)
        np.testing.assert_array_almost_equal(result, np.array([[[1.28922, 1.060341],
                                                                [1.201249, 0.974847]]]))
        np.testing.assert_array_almost_equal(result_error, np.array([[[0.121739, 0.168973],
                                                                      [0.111328, 0.153143]]]))

    def test_finds_best_fit_with_zeros_in_flux_and_not_uncertainty(self):
        energies = np.geomspace(1, 1e10, 23)
        true_A, true_gamma = 2.0, -1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0

        num_lats = 1
        num_lons = 1
        num_epochs = 1
        flux = np.array(flux_data).reshape(1, 1, 1, len(energies))
        variance = np.array(errors).reshape(1, 1, 1, len(energies))

        result, result_error = spectral_fit(num_epochs, num_lons, num_lats, flux, variance, energies)
        np.testing.assert_array_almost_equal(result, np.array(1.959853).reshape(1, 1, 1))
