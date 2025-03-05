import unittest

import numpy as np

from imap_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    filter_and_flatten_regress_parameters, regress, calculate_fit_temperature_density_velocity
from tests.test_helpers import get_test_data_path


class TestMomentsCalculation(unittest.TestCase):
    def test_compute_maxwellian_weight_factors_reproduces_heritage_results(self):
        corrected_counts = np.array([[[536.0, 10000.0, 1.2]]])
        weight_factors = compute_maxwellian_weight_factors(corrected_counts)

        np.testing.assert_array_almost_equal(weight_factors, [[[0.044041, 0.017845, 0.816500]]])

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
        velocity_vectors = np.loadtxt(get_test_data_path("swe/fake_velocity_vectors.csv"), delimiter=",",
                                      dtype=np.float64)
        weights = np.loadtxt(get_test_data_path("swe/fake_weights.csv"), delimiter=",", dtype=np.float64)
        yreg = np.loadtxt(get_test_data_path("swe/fake_yreg.csv"), delimiter=",", dtype=np.float64)

        regression_values, chisq = regress(velocity_vectors[:10], weights[:10], yreg[:10])

        np.testing.assert_array_almost_equal(regression_values,
                                             [18.546747,
                                              18.706356,
                                              2.018933,
                                              0.413322,
                                              1.666656,
                                              3.930964,
                                              -7627.713430,
                                              -18209.865922,
                                              -4261.624494,
                                              -0.665255])
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
        corrected_energy_bins = np.array([-1, 0, 3, 4, 5])
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

        core_breakpoint = 0
        core_halo_breakpoint = 4.5
        vectors, actual_weights, yreg = filter_and_flatten_regress_parameters(corrected_energy_bins, velocity_vectors,
                                                                              phase_space_density, weights,
                                                                              core_breakpoint, core_halo_breakpoint)

        np.testing.assert_array_equal(vectors, [[3, 0, 0], [4, 0, 0], [10, 0, 0], [10, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(actual_weights, [3, 1e-36, 10, 11, 12])
        np.testing.assert_array_equal(yreg, [np.log(3), -80.6, np.log(10), np.log(11), np.log(12)])
