import unittest

import numpy as np

from imap_processing.swe.l3.science.calculate_spacecraft_potential import piece_wise_model, find_breakpoints, \
    average_flux, compute_velocity_in_dsp_frame_km_s, compute_look_directions


class TestCalculateSpacecraftPotential(unittest.TestCase):
    def test_average_flux(self):
        flux_data = np.array([
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
        ])
        geometric_weights = [0.5, 0.25, 0.25, 0]
        result = average_flux(flux_data, geometric_weights)

        expected_result = [
            ((1 * 0.5 + 2 * 0.25 + 3 * 0.25) + (5 * 0.5 + 6 * 0.25 + 7 * 0.25) + (9 * 0.5 + 10 * 0.25 + 11 * 0.25)) / 3,
            ((13 * 0.5 + 14 * 0.25 + 15 * 0.25) + (17 * 0.5 + 18 * 0.25 + 19 * 0.25) + (
                    21 * 0.5 + 22 * 0.25 + 23 * 0.25)) / 3,

        ]
        np.testing.assert_almost_equal(result, expected_result)

    def test_look_direction(self):
        inst_az = np.array([[0, 90], [180, 270]])
        inst_el = np.array([-90, 0, 90])
        shape = (2, 2, 3, 3)
        expected_look_direction = np.array([
            [
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                [
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 0, 1]
                ]
            ],
            [
                [
                    [0, 0, -1],
                    [0, -1, 0],
                    [0, 0, 1]
                ],
                [
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, 0, 1]
                ]
            ]
        ])

        actual_look_direction = compute_look_directions(inst_el, inst_az)
        np.testing.assert_array_almost_equal(actual_look_direction, expected_look_direction)

    def test_compute_velocity(self):
        energy = np.array([1, 2])
        inst_el = np.array([0, 90])
        inst_az = np.array([[0], [90]])

        scalar_speeds = np.sqrt(energy * 1.602176634e-19 * 2 /
                                9.109_383_7139e-31) / 1000
        expected_velocity = np.array([
            [
                [
                    [0, -scalar_speeds[0], 0],
                    [0, 0, -scalar_speeds[0]],
                ],
            ],
            [
                [
                    [scalar_speeds[1], 0, 0],
                    [0, 0, -scalar_speeds[1]],
                ],
            ],
        ]
        )

        velocity = compute_velocity_in_dsp_frame_km_s(energy, inst_el, inst_az)
        np.testing.assert_array_almost_equal(velocity, expected_velocity)

    def test_compute_velocity_and_confirm_energy_calculation(self):
        energy = np.linspace(5, 2000, 24)
        inst_el = np.linspace(-90, 90, 7)
        rng = np.random.default_rng(20250219)
        inst_az = rng.random((24, 30)) * 360
        velocity = compute_velocity_in_dsp_frame_km_s(energy, inst_el, inst_az)
        scalar_velocity = np.linalg.norm(velocity, axis=-1)
        calculated_energy = 0.5 * 9.109_383_7139e-31 * np.square(scalar_velocity * 1000) / 1.602176634e-19

        np.testing.assert_almost_equal(calculated_energy[0], energy[0])
        np.testing.assert_almost_equal(calculated_energy[1], energy[1])
        self.assertEqual((24, 30, 7, 3), velocity.shape)
        print(velocity.max())

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
