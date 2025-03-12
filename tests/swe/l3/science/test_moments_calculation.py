import dataclasses
import math
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from imap_l3_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    filter_and_flatten_regress_parameters, regress, calculate_fit_temperature_density_velocity, rotate_temperature, \
    rotate_dps_vector_to_rtn, Moments, halotrunc, compute_density_scale
from tests.test_helpers import get_test_data_path


class TestMomentsCalculation(unittest.TestCase):
    def test_compute_maxwellian_weight_factors_reproduces_heritage_results(self):
        count_rates = np.array([[[536.0, 10000.0, 1.2]]])
        acquisition_duration = np.array([[[80000, 80000]]])
        weight_factors = compute_maxwellian_weight_factors(count_rates, acquisition_duration)

        first_weight = np.sqrt(536 * 80000) / (536 * 80000)
        second_weight = np.sqrt(10000.0 * 80000) / (10000.0 * 80000)
        third_weight = np.sqrt(1.2 * 80000) / (1.2 * 80000)

        np.testing.assert_array_almost_equal(weight_factors, [[[first_weight, second_weight, third_weight]]])

    def test_regress_reproduces_heritage_results_given_all_test_data(self):
        velocity_vectors = np.loadtxt(get_test_data_path("swe/fake_velocity_vectors.csv"), delimiter=",",
                                      dtype=np.float64)
        weights = np.loadtxt(get_test_data_path("swe/fake_weights.csv"), delimiter=",", dtype=np.float64)
        yreg = np.loadtxt(get_test_data_path("swe/fake_yreg.csv"), delimiter=",", dtype=np.float64)

        regression_values, chisq = regress(velocity_vectors, weights, yreg)

        np.testing.assert_array_almost_equal(regression_values,
                                             [5.924450,
                                              5.796693,
                                              5.877938,
                                              0.107305,
                                              -0.018571,
                                              0.045502,
                                              -140.856230,
                                              248.422738,
                                              -754.801277,
                                              -0.281777])

        self.assertAlmostEqual(75398.120454, chisq, places=6)

    def test_regress_reproduces_heritage_results_given_10_rows_of_test_data(self):
        velocity_vectors = np.loadtxt(get_test_data_path("swe/fake_velocity_vectors_10.csv"), delimiter=",",
                                      dtype=np.float64)
        weights = np.loadtxt(get_test_data_path("swe/fake_weights_10.csv"), delimiter=",", dtype=np.float64)
        yreg = np.loadtxt(get_test_data_path("swe/fake_yreg_10.csv"), delimiter=",", dtype=np.float64)

        regression_values, chisq = regress(velocity_vectors, weights, yreg)

        np.testing.assert_array_almost_equal(regression_values,
                                             [7.907190,
                                              8.474900,
                                              7.881836,
                                              -1.032437,
                                              -0.190888,
                                              -1.320381,
                                              202.408970,
                                              -98.152393,
                                              -1053.816409,
                                              0.074479])
        self.assertEqual(0, chisq)

    def test_calculate_fit_temperature_density_velocity_is_consistent_with_heritage_on_full_data(self):
        regress_output_of_full_fake_data = np.array([5.924450,
                                                     5.796693,
                                                     5.877938,
                                                     0.107305,
                                                     -0.018571,
                                                     0.045502,
                                                     -140.856230,
                                                     248.422738,
                                                     -754.801277,
                                                     -0.281777], dtype=np.float64)
        moments = calculate_fit_temperature_density_velocity(regress_output_of_full_fake_data)

        self.assertAlmostEqual(1.958299, moments.alpha, places=5)
        self.assertAlmostEqual(1.729684, moments.beta, places=5)
        self.assertAlmostEqual(0.167553, moments.t_perpendicular, places=5)
        self.assertAlmostEqual(0.176598, moments.t_parallel, places=5)
        self.assertAlmostEqual(0.000250, moments.velocity_x, places=5)
        self.assertAlmostEqual(-0.000443, moments.velocity_y, places=5)
        self.assertAlmostEqual(0.001288, moments.velocity_z, places=5)
        self.assertAlmostEqual(4.9549619600156776e10, moments.density, delta=2.5e4)
        self.assertAlmostEqual(112214.467052, moments.aoo, places=2)

    def test_calculate_fit_temperature_density_velocity_is_consistent_with_heritage_on_first_ten_vectors_data(self):
        regress_output_of_full_fake_data = np.array([18.546747,
                                                     18.706356,
                                                     2.018933,
                                                     0.413322,
                                                     1.666656,
                                                     3.930964,
                                                     -7627.713430,
                                                     -18209.865922,
                                                     -4261.624494,
                                                     -0.665255], dtype=np.float64)
        moments = calculate_fit_temperature_density_velocity(regress_output_of_full_fake_data)

        self.assertAlmostEqual(1.169789, moments.alpha, places=5)
        self.assertAlmostEqual(0.263119, moments.beta, places=5)
        self.assertAlmostEqual(0.054432, moments.t_perpendicular, places=5)
        self.assertAlmostEqual(0.395409, moments.t_parallel, places=5)
        self.assertAlmostEqual(0.004043, moments.velocity_x, places=5)
        self.assertAlmostEqual(0.010004, moments.velocity_y, places=5)
        self.assertAlmostEqual(-0.001708, moments.velocity_z, places=5)
        self.assertAlmostEqual(32905109580.985397, moments.density, delta=2.5e4)
        self.assertAlmostEqual(21193300.548418, moments.aoo, delta=1)

    def test_filter_and_flatten_regress_parameters(self):
        corrected_energy_bins = np.array([-1, 0, 3, 4.5, 5])
        phase_space_density = np.array([
            [[1, 2], [2, 3]],
            [[5, 6], [6, 7]],
            [[3, 1e-36], [0, 0]],
            [[10, 11], [0, 12]],
            [[21, 22], [23, 24]],
        ])

        weights = np.array([
            [[1, 2], [2, 3]],
            [[5, 6], [6, 7]],
            [[3, 1e-36], [0, 0]],
            [[10, 11], [0, 12]],
            [[21, 22], [23, 24]],
        ])

        velocity_vectors = np.array([
            [[[1, 0, 0], [1, 0, 0]], [[2, 0, 0], [2, 0, 0]]],
            [[[5, 0, 0], [5, 0, 0]], [[6, 0, 0], [6, 0, 0]]],
            [[[3, 0, 0], [4, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
            [[[10, 0, 0], [10, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
            [[[20, 0, 0], [8, 0, 0]], [[12, 0, 0], [23, 0, 0]]],
        ])

        core_breakpoint_index = 1
        core_halo_breakpoint_index = 4
        vectors, actual_weights, yreg = filter_and_flatten_regress_parameters(corrected_energy_bins, velocity_vectors,
                                                                              phase_space_density, weights,
                                                                              core_breakpoint_index,
                                                                              core_halo_breakpoint_index)

        np.testing.assert_array_equal(vectors, [[3, 0, 0], [4, 0, 0], [10, 0, 0], [10, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(actual_weights, [3, 1e-36, 10, 11, 12])
        np.testing.assert_array_equal(yreg, [np.log(3), -80.6, np.log(10), np.log(11), np.log(12)])

    @patch('spiceypy.spiceypy.pxform')
    @patch('spiceypy.spiceypy.datetime2et')
    def test_rotate_dps_vector_to_rtn(self, mock_datetime2et, mock_pxform):
        epoch = datetime(year=2020, month=3, day=10)
        dsp_vector = np.array([0, 1, 0])
        rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        mock_pxform.return_value = rotation_matrix

        rtn_vector = rotate_dps_vector_to_rtn(epoch, dsp_vector)
        mock_datetime2et.assert_called_once_with(epoch)

        mock_pxform.assert_called_once_with("IMAP_DPS", "IMAP_RTN", mock_datetime2et.return_value)

        np.testing.assert_array_equal(rtn_vector, rotation_matrix @ dsp_vector)

    @patch('spiceypy.spiceypy.pxform')
    @patch('spiceypy.spiceypy.datetime2et')
    def test_rotate_temperature(self, mock_datetime2et, mock_pxform):
        epoch = datetime(year=2020, month=3, day=11)
        temperature_alpha = math.pi / 4
        temperature_beta = math.pi / 8

        rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        mock_pxform.return_value = rotation_matrix

        theta, phi = rotate_temperature(epoch, temperature_alpha, temperature_beta)
        mock_datetime2et.assert_called_once_with(epoch)

        mock_pxform.assert_called_once_with("IMAP_DPS", "IMAP_RTN", mock_datetime2et.return_value)

        sin_dec = np.sin(temperature_beta)
        x = sin_dec * np.cos(temperature_alpha)
        y = sin_dec * np.sin(temperature_alpha)
        z = np.cos(temperature_beta)

        expected_rtn_temperature = rotation_matrix @ np.array([x, y, z])
        expected_rtn_temperature /= np.linalg.norm(expected_rtn_temperature)

        self.assertEqual(np.asin(expected_rtn_temperature[2]), phi)
        self.assertEqual(np.atan2(expected_rtn_temperature[1], expected_rtn_temperature[0]), theta)

    def test_momscale(self):
        core_halo_break = 125.6
        spacecraft_potential = 25.6

        test_cases = [
            ("normal values", 1.92424e4, 1.74908e5, 68.3, 3497.7912, 1e-4),
            ("the sun froze", 12301, 12301, 1500, 7.0965e62, 1e57),
            ("the sun froze", 12301, 12301, 400.3, 1.99655e+45, 1e40)
        ]

        for test_name, t_parallel, t_perpendicular, speed, expected_density_scale, allowable_difference in test_cases:
            with self.subTest(test_name):
                temperature = (t_parallel + 2 * t_perpendicular) / 3
                dscale = compute_density_scale(core_halo_break - spacecraft_potential, speed, temperature)

                self.assertAlmostEqual(expected_density_scale, dscale, delta=allowable_difference)

    def test_halotrunc(self):
        density = 130

        spacecraft_potential = 5

        moments = Moments(
            alpha=0,
            beta=0,
            t_parallel=0,
            t_perpendicular=0,
            velocity_x=102.3,
            velocity_y=94.9,
            velocity_z=86.7,
            density=density,
            aoo=0,
            ao=0,
        )

        test_cases = [
            ("t parallel less then 1e4, density should not change", 1e3, 1e7, 15, moments.density),
            ("t perpendicular less then 1e4, density should not change", 1e7, 1e2, 15, moments.density),
            ("t perpendicular greater then 1e8, density should be thrown out", 1e9, 1e1, 15, None),
            ("t parallel greater then 1e8, density should be thrown out", 1e2, 1e9, 15, None),
            ("core energy range is greater than 5 and temp is greater than 1e7", 1e7, 1.5e7, 15, moments.density),
            ("core energy range is less than 5 and temperatures greater than 1e4", 1e5, 1.5e6, 10, moments.density),
            ("core energy range is greater than 5 and temperatures greater than 1e4", 1e5, 1.5e6, 15, 126.487),
        ]

        for test_name, t_parallel, t_perpendicular, core_halo_breakpoint, expected_density in test_cases:
            with self.subTest(test_name):
                moments = dataclasses.replace(moments, t_parallel=t_parallel, t_perpendicular=t_perpendicular)

                scaled_density = halotrunc(moments, core_halo_breakpoint, spacecraft_potential)

                self.assertAlmostEqual(expected_density, scaled_density, places=3)
