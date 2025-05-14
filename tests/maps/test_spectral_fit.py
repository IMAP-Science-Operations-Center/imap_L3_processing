import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from imap_l3_processing.maps.map_models import IntensityMapData, RectangularIntensityMapData
from imap_l3_processing.maps.mpfit import mpfit
from imap_l3_processing.maps.spectral_fit import power_law, fit_arrays_to_power_law, fit_spectral_index_map, \
    calculate_spectral_index_for_multiple_ranges
from tests.test_helpers import get_test_data_path


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
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_equal(result, np.array(true_gamma).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array([0.060068]).reshape(1, 1, *spacial_dimension_shape))

    def test_spectral_fit_map(self):
        epoch = np.array([datetime.now()])
        energies = np.geomspace(1, 10, 23)
        true_A, true_gamma = 2.0, 1.5
        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)
        latitude = np.array([0])
        longitude = np.array([0])
        full_shape = (len(epoch), len(energies), len(longitude), len(latitude))

        data = IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([]),
            energy=energies,
            energy_delta_plus=np.repeat(0.5, 23),
            energy_delta_minus=np.repeat(0.5, 23),
            energy_label=np.array([]),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.full(full_shape, 1.0),
            obs_date=np.ma.array(np.full(full_shape, datetime(year=2010, month=1, day=1))),
            obs_date_range=np.full(full_shape, 100000),
            solid_angle=np.full(full_shape, 1.23),
            ena_intensity=np.array(flux_data).reshape(full_shape),
            ena_intensity_sys_err=np.array([]),
            ena_intensity_stat_unc=np.array(errors).reshape(full_shape)
        )

        spectral_intensity_map = fit_spectral_index_map(data)

        expected_energy_midpoint = np.sqrt(0.5 * 10.5)
        expected_energy_minus = expected_energy_midpoint - 0.5
        expected_energy_plus = 10.5 - expected_energy_midpoint
        expected_energy_label = f"0.5 - 10.5"
        np.testing.assert_array_equal(spectral_intensity_map.ena_spectral_index,
                                      np.array(true_gamma).reshape(1, 1, 1, 1))

        np.testing.assert_array_almost_equal(spectral_intensity_map.ena_spectral_index_stat_unc,
                                             np.array([0.060068]).reshape(1, 1, 1, 1))
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy, [expected_energy_midpoint])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_minus, [expected_energy_minus])
        np.testing.assert_array_almost_equal(spectral_intensity_map.energy_delta_plus, [expected_energy_plus])
        np.testing.assert_equal(spectral_intensity_map.energy_label, [expected_energy_label])

        np.testing.assert_array_almost_equal(spectral_intensity_map.longitude, data.longitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.latitude, data.latitude)
        np.testing.assert_array_almost_equal(spectral_intensity_map.solid_angle, data.solid_angle)

    def test_spectral_fit_fields_other_than_fit_fields(self):
        input_energies = np.array([1, 10, 99]) + 0.5
        input_deltas = np.array([0.5, 1, 0.5])

        latitude = np.arange(-90, 90, 45)
        longitude = np.arange(0, 360, 45)

        input_shape = (1, 3, len(longitude), len(latitude))
        input_map = IntensityMapData(
            epoch=np.array([datetime.now()]),
            epoch_delta=np.array([1000000]),
            energy=input_energies,
            energy_delta_plus=input_deltas,
            energy_delta_minus=input_deltas,
            energy_label=np.array(["a", "b", "c"]),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.zeros(input_shape),
            obs_date=np.full(input_shape, datetime(2025, 1, 1)),
            obs_date_range=np.full(input_shape, 1),
            solid_angle=np.full((len(longitude), len(latitude)), 0.1),
            ena_intensity=np.full(input_shape, 1),
            ena_intensity_sys_err=np.full(input_shape, 1),
            ena_intensity_stat_unc=np.full(input_shape, 1)
        )

        input_map.obs_date[0, 0] = datetime(2025, 1, 1)
        input_map.obs_date[0, 1] = datetime(2025, 1, 1)
        input_map.obs_date[0, 2] = datetime(2027, 1, 1)

        input_map.exposure_factor[0, 0] = 1.0
        input_map.exposure_factor[0, 1] = 2.0
        input_map.exposure_factor[0, 2] = 3.0

        input_map.obs_date_range[0, 0] = 1
        input_map.obs_date_range[0, 1] = 1
        input_map.obs_date_range[0, 2] = 3

        output = fit_spectral_index_map(input_map)

        self.assertEqual(output.energy.shape[0], 1)
        self.assertEqual(output.energy[0], 10)
        np.testing.assert_allclose(output.energy_delta_minus, np.array([9]))
        np.testing.assert_allclose(output.energy_delta_plus, np.array([90]))
        self.assertEqual(output.energy_label.shape, (1,))
        self.assertEqual("1.0 - 100.0", output.energy_label[0])

        expected_ena_shape = np.array([1, 1, len(longitude), len(latitude)])
        np.testing.assert_array_equal(output.obs_date,
                                      np.full(expected_ena_shape, datetime(2026, 1, 1)))
        np.testing.assert_array_equal(output.obs_date_range, np.full(expected_ena_shape, 2))
        np.testing.assert_array_equal(output.exposure_factor, np.full(expected_ena_shape, 6))

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
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
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
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
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
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
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

        result, result_error = fit_arrays_to_power_law(flux_data, errors, energies)
        np.testing.assert_array_almost_equal(result, np.array([[[[1.811566, 1.489658],
                                                                 [1.480259, 1.317993]]]]))
        np.testing.assert_array_almost_equal(result_error, np.array([[[[0.279224, 0.409522],
                                                                       [0.260374, 0.318162]]]]))

    def test_finds_best_fit_with_zeros_in_flux_and_not_uncertainty(self):
        energies = np.geomspace(1, 1e10, 23)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        errors = 0.2 * np.abs(flux_data)

        flux_data[0:3] = 0

        for name, spacial_dimension_shape in [("rectangular", (1, 1)), ("healpix", (1,))]:
            with self.subTest(name):
                flux = np.array(flux_data).reshape(1, len(energies), *spacial_dimension_shape)
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_almost_equal(result,
                                                     np.array([1.472697]).reshape(1, 1, *spacial_dimension_shape))

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
                uncertainty = np.array(errors).reshape(1, len(energies), *spacial_dimension_shape)

                result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
                np.testing.assert_array_almost_equal(result, np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))
                np.testing.assert_array_almost_equal(result_error,
                                                     np.array(np.nan).reshape(1, 1, *spacial_dimension_shape))

    def test_spectral_fit_can_fit_multiple_energy_ranges(self):
        epoch = np.array([datetime.now()])
        input_energy_range_1 = np.geomspace(1, 100, 11)
        input_energy_range_2 = np.geomspace(101, 10000, 12)
        true_A_range_1, true_gamma_range_1 = 2.0, 1.5
        true_A_range_2, true_gamma_range_2 = 0.1, 3.5
        flux_data_range_1 = true_A_range_1 * np.power(input_energy_range_1, -true_gamma_range_1)
        flux_data_range_2 = true_A_range_2 * np.power(input_energy_range_2, -true_gamma_range_2)
        longitude = np.array([45])
        latitude = np.array([90])
        errors_range_1 = 0.1 * np.abs(flux_data_range_1)
        errors_range_2 = 0.0001 * np.abs(flux_data_range_2)
        energies = np.append(input_energy_range_1, input_energy_range_2)
        full_shape = (len(epoch), len(energies), len(longitude), len(latitude))
        variance = np.concat((errors_range_1, errors_range_2)).reshape(full_shape)
        flux = np.concat((flux_data_range_1, flux_data_range_2)).reshape(full_shape)

        data = IntensityMapData(
            epoch=epoch,
            epoch_delta=np.array([10000]),
            energy=energies,
            energy_delta_plus=np.repeat(0.5, 23),
            energy_delta_minus=np.repeat(0.5, 23),
            energy_label=energies.astype(str),
            latitude=latitude,
            longitude=longitude,
            exposure_factor=np.full(full_shape, 1.0),
            obs_date=np.ma.array(np.full(full_shape, datetime(year=2010, month=1, day=1))),
            obs_date_range=np.full(full_shape, 100000),
            solid_angle=np.full(full_shape, 1.23),
            ena_intensity=flux,
            ena_intensity_sys_err=np.zeros_like(flux),
            ena_intensity_stat_unc=np.array(variance).reshape(full_shape)
        )

        output_energies = np.array([[1, 100.5], [100.5, 10000.5]])

        spectral_index_map_data = calculate_spectral_index_for_multiple_ranges(data, output_energies)
        np.testing.assert_array_equal(spectral_index_map_data.ena_spectral_index[0, 0, 0, 0], true_gamma_range_1)
        np.testing.assert_array_equal(spectral_index_map_data.ena_spectral_index[0, 1, 0, 0], true_gamma_range_2)
        np.testing.assert_array_almost_equal(spectral_index_map_data.ena_spectral_index_stat_unc[0, 0, 0, 0],
                                             0.020704)
        np.testing.assert_array_almost_equal(spectral_index_map_data.ena_spectral_index_stat_unc[0, 1, 0, 0],
                                             2.00179e-05)

        np.testing.assert_almost_equal(spectral_index_map_data.energy,
                                       np.array([7.088723439378913, 1002.5219448969683]))
        np.testing.assert_almost_equal(spectral_index_map_data.energy_delta_plus, np.array([93.4112766, 8997.9780551]))
        np.testing.assert_almost_equal(spectral_index_map_data.energy_delta_minus, np.array([6.5887234, 902.0219449]))
        np.testing.assert_array_equal(spectral_index_map_data.epoch, epoch)
        np.testing.assert_array_equal(spectral_index_map_data.epoch_delta, data.epoch_delta)
        np.testing.assert_array_equal(spectral_index_map_data.latitude, latitude)
        np.testing.assert_array_equal(spectral_index_map_data.longitude, longitude)
        np.testing.assert_array_equal(spectral_index_map_data.solid_angle, data.solid_angle)

        np.testing.assert_array_equal(spectral_index_map_data.energy_label, ["0.5 - 100.5", "100.5 - 10000.5"])

        np.testing.assert_array_equal(spectral_index_map_data.exposure_factor, np.reshape([11, 12], (1, 2, 1, 1)))
        np.testing.assert_array_equal(spectral_index_map_data.obs_date,
                                      np.full((1, 2, 1, 1), datetime(year=2010, month=1, day=1)))
        np.testing.assert_array_equal(spectral_index_map_data.obs_date_range, np.full((1, 2, 1, 1), 100000))

    @patch('imap_l3_processing.maps.spectral_fit.mpfit', wraps=mpfit)
    def test_passes_initial_guess_to_mpfit_based_on_line_between_first_and_last_points_in_log_space(self, mock_mpfit):
        energies = np.geomspace(10, 1e4, 6)
        true_A, true_gamma = 2.0, 1.5

        flux_data = true_A * np.power(energies, -true_gamma)
        flux_data[0] = 1e6
        flux_data[-1] = 1

        errors = 0.2 * np.abs(flux_data)

        flux = np.array(flux_data).reshape(1, len(energies), 1)
        uncertainty = np.array(errors).reshape(1, len(energies), 1)

        result, result_error = fit_arrays_to_power_law(flux, uncertainty, energies)
        self.assertEqual((8, 2), mock_mpfit.call_args.args[1])

    def test_spectral_fit_against_validation_data(self):
        test_cases = [
            ("hi45", RectangularIntensityMapData.read_from_path(
                get_test_data_path("hi/fake_l2_maps/hi45-6months.cdf")
            ).intensity_map_data,
             "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv")
        ]

        for name, input_data, expected_gamma_path, expected_sigma_path, in test_cases:
            with self.subTest(name):
                expected_gamma = np.loadtxt(get_test_data_path(expected_gamma_path), delimiter=",", dtype=str).T
                expected_gamma[expected_gamma == "NaN"] = "-1"
                expected_gamma = expected_gamma.astype(np.float64)
                expected_gamma[expected_gamma == -1] = np.nan

                expected_gamma_sigma = np.loadtxt(get_test_data_path(expected_sigma_path), delimiter=",",
                                                  dtype=str).T
                expected_gamma_sigma[expected_gamma_sigma == "NaN"] = "-1"
                expected_gamma_sigma = expected_gamma_sigma.astype(np.float64)
                expected_gamma_sigma[expected_gamma_sigma == -1] = np.nan

                output_data = calculate_spectral_index_for_multiple_ranges(input_data, [[0, np.inf]])

                np.testing.assert_allclose(output_data.ena_spectral_index[0, 0],
                                           expected_gamma, atol=1e-3)
                np.testing.assert_allclose(output_data.ena_spectral_index_stat_unc[0, 0],
                                           expected_gamma_sigma, atol=1e-3)
