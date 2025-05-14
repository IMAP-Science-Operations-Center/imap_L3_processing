import unittest

import numpy as np

from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralIndexDependencies
from imap_l3_processing.maps.spectral_fit_data_product import process_spectral_index
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3SpectralIndexDependencies
from tests.test_helpers import get_test_data_path


class TestSpectralFitDataProduct(unittest.TestCase):
    def test_spectral_fit_against_validation_data(self):
        test_cases = [
            ("hi45", ["hi/fake_l2_maps/hi45-6months.cdf"], "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv",
             "hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv", HiL3SpectralIndexDependencies),
            ("ultra", ['ultra/fake_ultra_map_data.cdf', 'ultra/imap_ultra_ulc-spx-energy-ranges_20250507_v000.txt'],
             'ultra/expected_ultra_gammas.csv', 'ultra/expected_ultra_gamma_sigmas.csv',
             UltraL3SpectralIndexDependencies)
        ]

        for name, input_file_paths, expected_gamma_path, expected_sigma_path, dependency_class in test_cases:
            with self.subTest(name):
                dependencies = dependency_class.from_file_paths(
                    *[get_test_data_path(input_file_path) for input_file_path in input_file_paths]
                )

                expected_gamma = np.loadtxt(get_test_data_path(expected_gamma_path), delimiter=",", dtype=str).T
                expected_gamma[expected_gamma == "NaN"] = "-1"
                expected_gamma = expected_gamma.astype(np.float64)
                expected_gamma[expected_gamma == -1] = np.nan

                expected_gamma_sigma = np.loadtxt(get_test_data_path(expected_sigma_path), delimiter=",",
                                                  dtype=str).T
                expected_gamma_sigma[expected_gamma_sigma == "NaN"] = "-1"
                expected_gamma_sigma = expected_gamma_sigma.astype(np.float64)
                expected_gamma_sigma[expected_gamma_sigma == -1] = np.nan

                output_data = process_spectral_index(dependencies)

                np.testing.assert_allclose(output_data.ena_spectral_index[0],
                                           expected_gamma, atol=1e-3)
                np.testing.assert_allclose(output_data.ena_spectral_index_stat_unc[0, 0],
                                           expected_gamma_sigma, atol=1e-3)
