import unittest

import numpy as np

from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralIndexDependencies
from imap_l3_processing.maps.spectral_fit_data_product import process_spectral_index
from tests.test_helpers import get_test_data_path


class TestSpectralFitDataProduct(unittest.TestCase):
    def test_spectral_fit_with_hi_data(self):
        dependencies = HiL3SpectralIndexDependencies.from_file_paths(
            hi_l3_path=get_test_data_path("hi/fake_l2_maps/hi45-6months.cdf")
        )

        expected_gamma_path = get_test_data_path("hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam.csv")
        expected_sigma_path = get_test_data_path("hi/validation/IMAP-Hi45_6months_4.0x4.0_fit_gam_sig.csv")
        expected_gamma = np.loadtxt(expected_gamma_path, delimiter=",", dtype=str).T
        expected_gamma[expected_gamma == "NaN"] = "-1"
        expected_gamma = expected_gamma.astype(np.float64)
        expected_gamma[expected_gamma == -1] = np.nan

        expected_gamma_sigma = np.loadtxt(expected_sigma_path, delimiter=",", dtype=str).T
        expected_gamma_sigma[expected_gamma_sigma == "NaN"] = "-1"
        expected_gamma_sigma = expected_gamma_sigma.astype(np.float64)
        expected_gamma_sigma[expected_gamma_sigma == -1] = np.nan

        output_data = process_spectral_index(dependencies)

        np.testing.assert_allclose(output_data.ena_spectral_index, [[expected_gamma]], atol=1e-3)
        np.testing.assert_allclose(output_data.ena_spectral_index_stat_unc, [[expected_gamma_sigma]], atol=1e-3)
