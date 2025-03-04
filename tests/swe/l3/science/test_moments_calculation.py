import unittest

import numpy as np

from imap_processing.swe.l3.science.moment_calculations import compute_maxwellian_weight_factors, \
    filter_and_flatten_regress_parameters, regress
from tests.test_helpers import get_test_data_path


class TestMomentsCalculation(unittest.TestCase):
    def test_compute_maxwellian_weight_factors_reproduces_heritage_results(self):
        corrected_counts = np.array([[[536.0, 10000.0, 1.2]]])
        weight_factors = compute_maxwellian_weight_factors(corrected_counts)

        np.testing.assert_array_almost_equal(weight_factors, [[[0.044041, 0.017845, 0.816500]]])

    def test_regress_reproduces_heritage_results(self):
        velocity_vectors = np.loadtxt(get_test_data_path("swe/fake_velocity_vectors.csv"), delimiter=",")
        weights = np.loadtxt(get_test_data_path("swe/fake_weights.csv"), delimiter=",")
        yreg = np.loadtxt(get_test_data_path("swe/fake_yreg.csv"), delimiter=",")

        regression_values = regress(velocity_vectors, weights, yreg)

        np.testing.assert_array_almost_equal(regression_values,
                                             [5.741342, 5.938431, 5.829471, 0.029683, -0.063667, 0.000497, -215.541543,
                                              37.755161, -748.104342,
                                              -0.278396])

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
