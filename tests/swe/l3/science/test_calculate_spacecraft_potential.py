import unittest

import numpy as np

from imap_processing.swe.l3.science.calculate_spacecraft_potential import piece_wise_model, find_breakpoints


class TestCalculateSpacecraftPotential(unittest.TestCase):
    def test_find_breakpoints_with_noisy_data(self):
        xs = np.array([2.6600000e+00, 3.7050000e+00, 5.1300000e+00, 7.1725000e+00,
                       9.9750000e+00, 1.3870000e+01, 1.9285000e+01, 2.6790000e+01,
                       3.7287500e+01, 5.1870000e+01, 7.2152500e+01, 1.0036750e+02,
                       1.3960250e+02, 1.9418000e+02, 2.7013250e+02, 3.7572500e+02,
                       5.2264250e+02, 7.2698750e+02, 1.0112275e+03, 1.4066650e+03])
        avg_flux = np.array([3.57482993e+01, 3.06214254e+01, 2.21006219e+01, 1.68925625e+01,
                             1.40040578e+01, 1.10364953e+01, 8.05239700e+00, 5.48587782e+00,
                             3.32793768e+00, 1.72978233e+00, 9.43240260e-01, 5.82430995e-01,
                             3.69484446e-01, 2.19359553e-01, 1.19059738e-01, 5.64115725e-02,
                             2.30604686e-02, 9.14406238e-03, 4.24754874e-03, 1.61814681e-03])
        spacecraft_potential, core_halo_breakpoint = find_breakpoints(xs, avg_flux)
        self.assertAlmostEqual(11.1, spacecraft_potential, 1)
        self.assertAlmostEqual(81.1, core_halo_breakpoint, 1)

    def test_find_breakpoints_with_synthetic_data(self):
        cases = [
            (7, 60),
            (15, 100),
            (11, 78),
        ]
        for case in cases:
            with self.subTest(case):
                expected_potential, expected_core_halo = case
                xs = np.array([2.6600000e+00, 3.7050000e+00, 5.1300000e+00, 7.1725000e+00,
                               9.9750000e+00, 1.3870000e+01, 1.9285000e+01, 2.6790000e+01,
                               3.7287500e+01, 5.1870000e+01, 7.2152500e+01, 1.0036750e+02,
                               1.3960250e+02, 1.9418000e+02, 2.7013250e+02, 3.7572500e+02,
                               5.2264250e+02, 7.2698750e+02, 1.0112275e+03, 1.4066650e+03])
                avg_flux = np.exp(piece_wise_model(xs, 1e4, 0.05, expected_potential, 0.02, expected_core_halo, 0.01))
                noise_floor = 1
                avg_flux += noise_floor
                spacecraft_potential, core_halo_breakpoint = find_breakpoints(xs, avg_flux)
                self.assertAlmostEqual(expected_potential, spacecraft_potential, 2)
                self.assertAlmostEqual(expected_core_halo, core_halo_breakpoint, 0)


if __name__ == '__main__':
    unittest.main()
