import unittest
from unittest.mock import Mock, patch

import numpy as np

from imap_processing.swe.l3.science.pitch_calculations import piece_wise_model, find_breakpoints, \
    average_flux, calculate_velocity_in_dsp_frame_km_s, calculate_look_directions, rebin_by_pitch_angle, \
    correct_and_rebin, calculate_energy_in_ev_from_velocity_in_km_per_second
from tests.test_helpers import build_swe_configuration


class TestPitchCalculations(unittest.TestCase):
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

        actual_look_direction = calculate_look_directions(inst_el, inst_az)
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

        velocity = calculate_velocity_in_dsp_frame_km_s(energy, inst_el, inst_az)
        np.testing.assert_array_almost_equal(velocity, expected_velocity)

    def test_compute_velocity_and_confirm_energy_calculation(self):
        energy = np.linspace(5, 2000, 24)
        inst_el = np.linspace(-90, 90, 7)
        rng = np.random.default_rng(20250219)
        inst_az = rng.random((24, 30)) * 360
        velocity = calculate_velocity_in_dsp_frame_km_s(energy, inst_el, inst_az)
        scalar_velocity = np.linalg.norm(velocity, axis=-1)
        calculated_energy = 0.5 * 9.109_383_7139e-31 * np.square(scalar_velocity * 1000) / 1.602176634e-19

        np.testing.assert_almost_equal(calculated_energy[0], energy[0])
        np.testing.assert_almost_equal(calculated_energy[1], energy[1])
        self.assertEqual((24, 30, 7, 3), velocity.shape)

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

    def test_rebin_by_pitch_angle(self):
        flux = np.array([1000, 10, 32, 256])
        pitch_angle = np.array([25, 60, 120, 170])
        energy = np.array([10 * 0.8, 10 / 0.8, 10 * 0.9 * 0.9, 10 / 0.9])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_delta=[45, 45],
            energy_bins=[10]
        )

        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array([
            [100, 128]
        ]
        )
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_uses_energy_within_60_to_140_percent_of_nominal(self):
        flux = np.array([9999, 50, 50, 9999])
        pitch_angle = np.array([25, 60, 120, 170])
        energy = np.array([10 * 0.59, 10 * 0.61, 10 * 1.39, 10 * 1.41])

        config = build_swe_configuration(
            pitch_angle_bins=[90],
            pitch_angle_delta=[90],
            energy_bins=[10]
        )
        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array([
            [50]
        ]
        )
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_skips_bins_with_less_than_two_measurements(self):
        flux = np.array([9999, 50, 50, 9999])
        pitch_angle = np.array([91, 95, 120, 170])
        energy = np.array([10 * 0.59, 10 * 0.61, 10 * 1.39, 50 * 1.1])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_delta=[45, 45],
            energy_bins=[10, 50]
        )

        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array(
            [
                [np.nan, 50],
                [np.nan, np.nan]
            ]
        )
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_full_size_data(self):
        rng = np.random.default_rng(202502201113)
        flux = rng.random((24, 30, 7)) * 1000
        pitch_angle = rng.random((24, 30, 7)) * 180
        energy = rng.random((24, 30, 7)) * 1000
        pitch_angle_bins = np.linspace(0, 180, 20, endpoint=False) + 4.5
        pitch_angle_deltas = np.repeat(4.5, 20)
        energy_bins = np.geomspace(2, 5000, 24)
        config = build_swe_configuration(pitch_angle_bins=pitch_angle_bins, pitch_angle_delta=pitch_angle_deltas,
                                         energy_bins=energy_bins)
        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)
        self.assertEqual((24, 20), result.shape)

    def test_calculate_energy(self):
        velocities = np.array([
            [400, 0, 0],
            [0, -25000, 0],
        ])
        expected_energies = [0.45485041, 1776.7594]
        result = calculate_energy_in_ev_from_velocity_in_km_per_second(velocities)
        np.testing.assert_allclose(result, expected_energies, rtol=1e-7)

    @patch('imap_processing.swe.l3.science.pitch_calculations.calculate_energy_in_ev_from_velocity_in_km_per_second')
    @patch('imap_processing.swe.l3.science.pitch_calculations.rebin_by_pitch_angle')
    @patch('imap_processing.swe.l3.science.pitch_calculations.calculate_pitch_angle')
    @patch('imap_processing.swe.l3.science.pitch_calculations.calculate_velocity_in_sw_frame')
    @patch('imap_processing.swe.l3.science.pitch_calculations.calculate_velocity_in_dsp_frame_km_s')
    def test_correct_and_rebin(self, mock_calculate_dsp_velocity, mock_calculate_velocity_in_sw_frame,
                               mock_calculate_pitch_angle, mock_rebin_by_pitch_angle, mock_calculate_energy):
        flux_data = Mock()
        corrected_energy = Mock()
        inst_el = Mock()
        inst_az = Mock()
        mag_vector = Mock()
        solar_wind_vector = Mock()
        configuration = Mock()
        result = correct_and_rebin(
            flux_or_psd=flux_data,
            energy_bins_minus_potential=corrected_energy,
            inst_el=inst_el, inst_az=inst_az,
            mag_vector=mag_vector, solar_wind_vector=solar_wind_vector,
            config=configuration,
        )

        mock_calculate_dsp_velocity.assert_called_once_with(corrected_energy, inst_el, inst_az)
        mock_calculate_velocity_in_sw_frame.assert_called_once_with(
            mock_calculate_dsp_velocity.return_value, solar_wind_vector)
        mock_calculate_pitch_angle.assert_called_once_with(mock_calculate_velocity_in_sw_frame.return_value, mag_vector)
        mock_calculate_energy.assert_called_with(mock_calculate_velocity_in_sw_frame.return_value)
        mock_rebin_by_pitch_angle.assert_called_with(
            flux_data, mock_calculate_pitch_angle.return_value,
            mock_calculate_energy.return_value,
            configuration,
        )
        self.assertEqual(mock_rebin_by_pitch_angle.return_value, result)


if __name__ == '__main__':
    unittest.main()
