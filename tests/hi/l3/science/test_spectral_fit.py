import unittest
from unittest.mock import patch

import numpy as np

from imap_l3_processing.hi.l3.science.mpfit import mpfit
from imap_l3_processing.hi.l3.science.spectral_fit import power_law, spectral_fit


class TestSpectralFit(unittest.TestCase):
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
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array([0.231729]).reshape(1, 1, *spacial_dimension_shape))

    def test_finds_best_fit_with_nan_in_flux(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        flux_data[len(flux_data) // 2] = np.nan
        flux_data[0] = np.nan
        flux_data[-1] = np.nan

        errors = 0.2 * np.abs(flux_data)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))

    def test_finds_best_fit_with_nan_in_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)

        errors = 0.2 * np.abs(flux_data)
        errors[len(errors) // 2] = np.nan
        errors[0] = np.nan
        errors[-1] = np.nan

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))

    def test_finds_best_fit_with_zero_in_flux_and_uncertainty(self):
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0
        errors[0:3] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))

    def test_finds_best_fit_with_ibex_data(self):
        energies = np.array([0.71, 1.11, 1.74, 2.73, 4.29])
        flux_data = np.array([[[[46.710853, 61.45169],
                                [60.682266, 60.523616]],
                               [[34.896639, 11.987692],
                                [29.323209, 19.239248]],
                               [[18.899919, 17.140896],
                                [16.12169, 14.099767]],
                               [[14.709325, 12.165362],
                                [13.114869, 13.726924]],
                               [[5.815468, 5.071331],
                                [6.060111, 6.499908]]]])
        errors = np.array([[[[4.17239236e+02, 5.87577556e+02],
                             [4.59947913e+02, 7.02472212e+02]],
                            [[6.38942760e+01, 3.62103490e+01],
                             [2.87564900e+01, 5.06517210e+01]],
                            [[1.58932960e+01, 1.41284380e+01],
                             [1.34992710e+01, 9.66067900e+00]],
                            [[2.01988700e+00, 2.75456600e+00],
                             [1.83946100e+00, 2.25438200e+00]],
                            [[3.00438000e-01, 3.76740000e-01],
                             [2.15943000e-01, 3.76271000e-01]]]])

        result, result_error = spectral_fit(flux_data, errors, energies)
        np.testing.assert_array_almost_equal(result, np.array([[[[1.28922, 1.060341],
                                                                 [1.201249, 0.974847]]]]))
        np.testing.assert_array_almost_equal(result_error, np.array([[[[0.121739, 0.168973],
                                                                       [0.111328, 0.153143]]]]))

    def test_finds_best_fit_with_zeros_in_flux_and_not_uncertainty(self):
        energies = np.geomspace(1, 1e10, 23)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_almost_equal(result, np.array(0.861911).reshape(1, 1, *spacial_dimension_shape))

    def test_returns_nan_when_only_one_point_is_valid(self):
        energies = np.geomspace(1, 1e10, 5)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[1:] = 0
        errors[1:] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                variance = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance, energies)
                np.testing.assert_array_almost_equal(result, np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))

    def test_spectral_fit_can_fit_multiple_energy_ranges(self):
        input_energy_range_1 = np.geomspace(1, 100, 11)
        input_energy_range_2 = np.geomspace(101, 10000, 12)
        true_A_range_1, true_gamma_range_1 = 2.0, 1.5
        true_A_range_2, true_gamma_range_2 = 0.1, 3.5
        flux_data_range_1 = true_A_range_1 * np.power(input_energy_range_1, -true_gamma_range_1)
        flux_data_range_2 = true_A_range_2 * np.power(input_energy_range_2, -true_gamma_range_2)

        errors_range_1 = 0.1 * np.abs(flux_data_range_1)
        errors_range_2 = 0.0001 * np.abs(flux_data_range_2)

        cases = [
            ("rectangular", (1, 1)),
            ("healpix", (1,))
        ]

        output_energies = np.array([[1, 100.5], [100.5, 10000]])

        for name, spacial_dimension_shape in cases:
            with self.subTest(name):
                flux = np.concat((flux_data_range_1, flux_data_range_2)) \
                    .reshape(1, len(input_energy_range_1) + len(input_energy_range_2), *spacial_dimension_shape)
                variance = np.concat((errors_range_1, errors_range_2)) \
                    .reshape(1, len(input_energy_range_1) + len(input_energy_range_2), *spacial_dimension_shape)

                result, result_error = spectral_fit(flux, variance,
                                                    np.concat((input_energy_range_1, input_energy_range_2)),
                                                    output_energies)

                result_range_1 = result[0][0]
                result_range_2 = result[0][1]

                np.testing.assert_array_equal(result_range_1,
                                              np.array(true_gamma_range_1).reshape(*spacial_dimension_shape))
                np.testing.assert_array_equal(result_range_2,
                                              np.array(true_gamma_range_2).reshape(*spacial_dimension_shape))

    @patch('imap_l3_processing.hi.l3.science.spectral_fit.mpfit', wraps=mpfit)
    def test_passes_initial_guess_to_mpfit_based_on_line_between_first_and_last_points_in_log_space(self, mock_mpfit):
        energies = np.geomspace(10, 1e4, 6)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        flux_data[0] = 1e6
        flux_data[-1] = 1

        errors = 0.2 * np.abs(flux_data)

        flux = np.array(flux_data).reshape(1, len(energies), 1)
        variance = np.array(errors).reshape(1, len(energies), 1)

        result, result_error = spectral_fit(flux, variance, energies)
        self.assertEqual((8, 2), mock_mpfit.call_args.args[1])
