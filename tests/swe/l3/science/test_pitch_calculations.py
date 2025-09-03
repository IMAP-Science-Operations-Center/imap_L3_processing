import unittest
from unittest.mock import Mock, patch, ANY, call, sentinel

import numpy as np

from imap_l3_processing.swe.l3.science.pitch_calculations import find_breakpoints, \
    average_over_look_directions, calculate_velocity_in_dsp_frame_km_s, calculate_look_directions, rebin_by_pitch_angle, \
    correct_and_rebin, calculate_energy_in_ev_from_velocity_in_km_per_second, integrate_distribution_to_get_1d_spectrum, \
    integrate_distribution_to_get_inbound_and_outbound_1d_spectrum, try_curve_fit_until_valid, \
    rebin_by_pitch_angle_and_gyrophase, swe_rebin_intensity_by_pitch_angle_and_gyrophase, ls_fit
from tests.test_helpers import build_swe_configuration, NumpyArrayMatcher


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
        result = average_over_look_directions(flux_data, geometric_weights, 1e-32)

        expected_result = [
            ((1 * 0.5 + 2 * 0.25 + 3 * 0.25) + (5 * 0.5 + 6 * 0.25 + 7 * 0.25) + (9 * 0.5 + 10 * 0.25 + 11 * 0.25)) / 3,
            ((13 * 0.5 + 14 * 0.25 + 15 * 0.25) + (17 * 0.5 + 18 * 0.25 + 19 * 0.25) + (
                    21 * 0.5 + 22 * 0.25 + 23 * 0.25)) / 3,

        ]
        np.testing.assert_almost_equal(result, expected_result)

    def test_average_over_look_directions_with_zeroes(self):
        flux_data = np.array([
            [
                [1, 2, 3, 4],
            ],
            [
                [0, 0, 0, 1e-36],
            ],
        ])
        geometric_weights = [0.25, 0.25, 0.25, 0.25]
        result = average_over_look_directions(flux_data, geometric_weights, 1e-34)

        expected_result = [
            (1 * 0.25 + 2 * 0.25 + 3 * 0.25 + 4 * 0.25), 1e-34
        ]
        np.testing.assert_allclose(result, expected_result)

    def test_look_direction(self):
        inst_az = np.array([[0, 90], [180, 270]])
        inst_el = np.array([-90, 0, 90])
        shape = (2, 2, 3, 3)
        expected_look_direction = np.array([
            [
                [
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, 0, 1]
                ],
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            ],
            [
                [
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 0, 1]
                ],
                [
                    [0, 0, -1],
                    [0, -1, 0],
                    [0, 0, 1]
                ]
            ]
        ])

        actual_look_direction = calculate_look_directions(inst_el, inst_az)
        np.testing.assert_array_almost_equal(actual_look_direction, expected_look_direction)

    def test_compute_velocity(self):
        energy = np.array([1, 2])  # 2 -- energy
        inst_el = np.array([0, 90])
        inst_az = np.array([[0], [90]])  # 2, 1  --  energy, spin sector

        scalar_speeds = np.sqrt(energy * 1.602176634e-19 * 2 /
                                9.109_383_7139e-31) / 1000
        expected_velocity = np.array([
            [  # energy 1
                [  # spin sector 1
                    [-scalar_speeds[0], 0, 0],  # CEM 1
                    [0, 0, -scalar_speeds[0]],
                ],
            ],
            [  # energy 2
                [
                    [0, -scalar_speeds[1], 0],
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

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.try_curve_fit_until_valid')
    def test_find_breakpoints_determines_b_deltas_correctly(self, mock_try_curve_fit_until_valid):
        config = build_swe_configuration(refit_core_halo_breakpoint_index=4, slope_ratio_cutoff_for_potential_calc=0)

        cases = [
            ("slope local max is on left side of data", [0.1, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03], -1.5, -10),
            ("slope local max in on right side of data", [0.1, 0.18, 0.15, 0.1, 0.09, 0.08, 0.12, 0.1], -1, 10),
            ("leftmost slope ratio local min > rightmost slope local max", [0.5, 0.6, 0.7, 0.8, 0.9, 0.75, 0.4, 0.35],
             -1.5, -10),
            ("no local min ratio or local max", [.8, .7, .6, .5, .4, .3, .2, .1], -1.5, -10),
            ("no local min ratio or local max", [.1, .2, .3, .4, .5, .6, .7, .8], -1.5, -10)
        ]
        for name, slopes, expected_b2_delta, expected_b4_delta in cases:
            with self.subTest(name):
                xs = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80])
                energy_deltas = np.diff(xs)
                initial = 10000

                diff_log_flux = -np.array(slopes) * energy_deltas
                log_flux = np.cumsum(np.append(np.log(initial), diff_log_flux))
                avg_flux = np.exp(log_flux)

                result = find_breakpoints(
                    xs, avg_flux, [10, 10, 10], [80, 80, 80], config)
                mock_try_curve_fit_until_valid.assert_called_with(ANY, ANY, ANY, 10, 80, expected_b2_delta,
                                                                  expected_b4_delta)

                self.assertEqual(mock_try_curve_fit_until_valid.return_value, result)

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.ls_fit')
    def test_find_breakpoints_uses_config_for_slope_guesses(self, ls_fit):
        ls_fit.return_value = [1, 3, 10, 2, 80, 1]

        cases = [
            (20, 100, 0.2, 0.3, 0.2),
            (100, 400, 0.2, 0.2, 0.15),
        ]
        for case in cases:
            with self.subTest(case):
                core_energy, halo_energy, b1, b3, b5 = case
                config = build_swe_configuration(
                    core_energy_for_slope_guess=core_energy,
                    halo_energy_for_slope_guess=halo_energy,
                )

                xs = np.array([1, 10, 50, 200, 800, 2400, 7200])
                slopes = np.array([0.2, 0.3, 0.2, 0.15, 0.1, 0.08])
                energy_deltas = np.diff(xs)
                initial = 10000

                diff_log_flux = -slopes * energy_deltas
                log_flux = np.cumsum(np.append(np.log(initial), diff_log_flux))

                avg_flux = np.exp(log_flux)

                spacecraft_potential, core_halo_breakpoint = find_breakpoints(
                    xs, avg_flux, [10, 10, 10], [80, 80, 80],
                    config)

                expected_guesses = [ANY, b1, 10, b3, 80, b5]
                rounded_actuals = [round(x, 6) for x in ls_fit.call_args.args[2]]
                self.assertEqual(expected_guesses, rounded_actuals)

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.ls_fit')
    def test_find_breakpoints_uses_config_for_slope_ratio(self, mock_ls_fit):
        mock_ls_fit.return_value = [1, 2, 3, 4, 5, 6]
        cases = [
            (0.55, 4),
            (0.45, 5),
            (0.35, 6),
        ]
        for case in cases:
            with self.subTest(case):
                cutoff, data_length = case
                config = build_swe_configuration(
                    slope_ratio_cutoff_for_potential_calc=cutoff
                )

                xs = np.arange(10)
                slope_ratios = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
                slopes = np.cumprod(np.append(0.1, slope_ratios))
                energy_deltas = np.diff(xs)
                initial = 10000
                diff_log_flux = -slopes * energy_deltas
                log_flux = np.cumsum(np.append(np.log(initial), diff_log_flux))
                avg_flux = np.exp(log_flux)

                spacecraft_potential, core_halo_breakpoint = find_breakpoints(
                    xs, avg_flux, [10, 10, 10], [80, 80, 80], config)

                np.testing.assert_almost_equal(mock_ls_fit.call_args.args[0], xs[:data_length])
                np.testing.assert_almost_equal(mock_ls_fit.call_args.args[1], log_flux[:data_length])

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.ls_fit')
    def test_try_curve_fit_until_valid(self, mock_ls_fit):
        cases = [
            ("happy case", [1, 3, 10, 2, 80, 1], 1, 15),
            ("b[1] <= 0", [1, -1, 10, 2, 80, 3], 2, 15),
            ("b[3] <= 0", [1, 3, 10, -1, 80, 3], 2, 15),
            ("b[5] <= 0", [1, 3, 10, 2, 80, -1], 2, 15),
            ("b[2] >= b[4]", [1, 15, 18, 30, 16, 100], 2, 15),
            ("b[4] <= 15", [1, 3, 10, 2, 12, 1], 2, 15),
            ("b[2] <= energies[0]", [1, 3, 0.8, 2, 80, 1], 2, 15),
            ("b[2] >= 20", [1, 3, 25, 2, 80, 1], 2, 15),
            ("b[2] >= 2x spacecraft potential", [1, 3, 15, 2, 80, 1], 2, 6),
        ]

        for name, curve_fit_first_result, call_count, latest_spacecraft_potential in cases:
            with self.subTest(name):
                mock_ls_fit.reset_mock()
                mock_ls_fit.side_effect = [
                    curve_fit_first_result,
                    [1, 3, 10, 2, 80, 1]
                ]
                energies = [1, 10, 20, 50, 100, 250]
                log_flux = [.1, 1, 10, 100, 1000, 10000]
                initial_guesses = (0, 1, 2, 3, 4, 5)
                latest_core_halo_breakpoint = 10
                delta_b2 = -1
                delta_b4 = 10

                returned_fit = try_curve_fit_until_valid(energies, log_flux, initial_guesses,
                                                         latest_spacecraft_potential, latest_core_halo_breakpoint,
                                                         delta_b2,
                                                         delta_b4)
                self.assertEqual(call_count, mock_ls_fit.call_count)
                self.assertEqual(call(energies, log_flux, initial_guesses),
                                 mock_ls_fit.call_args_list[0])
                if call_count > 1:
                    modified_guesses = [0, 1, 2 + delta_b2, 3, 4 + delta_b4, 5]
                    self.assertEqual(call(energies, log_flux, modified_guesses),
                                     mock_ls_fit.call_args_list[1])
                self.assertEqual((10, 80), returned_fit)

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.ls_fit')
    def test_try_curve_fit_until_valid_tries_up_to_3_times(self, mock_ls_fit):
        good_fit = ([1, 3, 10, 2, 80, 1])
        bad_fit = ([1, 3, -20, 2, 80, 1])
        cases = [
            ("passes without changing b values", 1, [good_fit], 2, 4, (10, 80)),
            ("passes after 1 loop", 2, [bad_fit, good_fit], 1, 14, (10, 80)),
            ("passes after 2 loop", 3, [bad_fit, bad_fit, good_fit], 0, 24, (10, 80)),
            ("passes after 3 loop", 4, [bad_fit, bad_fit, bad_fit, good_fit], -1, 34, (10, 80)),
            ("does not pass after 3 loops", 4, [bad_fit, bad_fit, bad_fit, bad_fit], -1, 34, (15, 75)),
        ]
        for name, call_count, curve_fit_return_values, \
                expected_last_b2_guess, expected_last_b4_guess, \
                expected_result in cases:
            with self.subTest(name):
                mock_ls_fit.reset_mock()
                mock_ls_fit.side_effect = curve_fit_return_values
                energies = [1, 10, 20, 50, 100, 250]
                log_flux = [.1, 1, 10, 100, 1000, 10000]
                initial_guesses = (0, 1, 2, 3, 4, 5)
                delta_b2 = -1
                delta_b4 = 10

                result = try_curve_fit_until_valid(energies, log_flux, initial_guesses,
                                                   15, 75, delta_b2,
                                                   delta_b4)
                self.assertEqual(call_count, mock_ls_fit.call_count)
                last_guesses = mock_ls_fit.call_args.args[2]
                self.assertEqual(last_guesses[2], expected_last_b2_guess)
                self.assertEqual(last_guesses[4], expected_last_b4_guess)
                self.assertEqual(expected_result, result)

    def test_rebin_by_pitch_angle(self):
        flux = np.array([1000, 10, 32, 256])
        pitch_angle = np.array([25, 60, 120, 170])
        energy = np.array([10 * 0.8, 10 / 0.8, 10 * 0.9 * 0.9, 10 / 0.9])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_deltas=[45, 45],
            energy_bins=[10]
        )

        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array(
            [
                [100, 128]
            ]
        )
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_and_gyrophase(self):
        psd = np.array([1000, 10,
                        32, 256,
                        3, 9, 81,
                        5, 25, 125])

        pitch_angle = np.array([25, 60, 25, 60, 120, 170, 165, 120, 170, 165])
        gyrophase = np.array([10, 120, 210, 350, 10, 120, 95, 210, 350, 260])

        central_energy_value = 10  # bin range is really 60 - 140
        energy = np.array([central_energy_value * 0.8, central_energy_value / 0.8,
                           central_energy_value * 0.8 ** 2, central_energy_value / 0.8,
                           central_energy_value * 0.8 ** 2, central_energy_value * 0.8, central_energy_value / 0.8,
                           central_energy_value * 0.8, central_energy_value, central_energy_value / 0.8])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_deltas=[45, 45],
            energy_bins=[central_energy_value],
            gyrophase_bins=[90, 270],
            gyrophase_deltas=[90, 90]
        )

        rebinned_by_gyro = rebin_by_pitch_angle_and_gyrophase(psd, pitch_angle, gyrophase, energy, config)

        expected_gyro = np.array([
            [
                [100, 128],
                [27, 25]
            ]
        ])

        np.testing.assert_almost_equal(expected_gyro, rebinned_by_gyro)

    def test_rebin_by_pitch_angle_ignores_zero_measurements(self):
        flux = np.array([1000, 10, 32, 256, 0])
        pitch_angle = np.array([25, 60, 120, 170, 165])
        energy = np.array([10 * 0.8, 10 / 0.8, 10 * 0.9 * 0.9, 10 / 0.9, 10 / 0.9])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_deltas=[45, 45],
            energy_bins=[10]
        )

        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array([
            [100, 128]
        ])
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_uses_energy_within_configured_percent_of_nominal(self):
        flux = np.array([9999, 50, 50, 9999])
        pitch_angle = np.array([25, 60, 120, 170])
        energy = np.array([10 * 0.69, 10 * 0.71, 10 * 1.49, 10 * 1.51])

        config = build_swe_configuration(
            pitch_angle_bins=[90],
            pitch_angle_deltas=[90],
            energy_bins=[10],
            energy_bin_low_multiplier=0.7,
            energy_bin_high_multiplier=1.5,
        )
        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array([
            [50]
        ])
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_skips_bins_with_less_than_two_measurements(self):
        flux = np.array([9999, 50, 50, 9999])
        pitch_angle = np.array([91, 95, 120, 170])
        energy = np.array([10 * 0.59, 10 * 0.61, 10 * 1.39, 50 * 1.1])

        config = build_swe_configuration(
            pitch_angle_bins=[45, 135],
            pitch_angle_deltas=[45, 45],
            energy_bins=[10, 50],
            energy_bin_low_multiplier=0.6,
            energy_bin_high_multiplier=1.4,
        )

        result = rebin_by_pitch_angle(flux, pitch_angle, energy, config)

        expected_result = np.array(
            [
                [np.nan, 50],
                [np.nan, np.nan]
            ]
        )
        np.testing.assert_almost_equal(result, expected_result)

    def test_rebin_by_pitch_angle_skips_bins_with_invalid_measurements(self):
        test_cases = [
            ('no energy is close enough', [5.9, 6, 8.2, 11.8, 12.2], np.nan),
            ('overall max value is in range and an energy is close enough below', [8, 8.21, 11.8, 12, 12.1], 50),
            ('overall max value is in range and an energy is close enough above', [8, 8.20, 11.79, 12, 12.1], 50),
            ('min and max are outside window and points on both side of nominal', [5.8, 5.9, 14.1, 9, 11], 50),
            ('two lowest mins and max are outside window and points only below', [5.8, 5.9, 14.1, 8, 9], np.nan),
            ('two lowest mins and max are outside window and points only above', [5.8, 5.9, 14.1, 11, 12], np.nan),
            ('second_lowest in window and an energy close enough', [5.8, 6.1, 14.1, 11, 12], 50),
            ('second_lowest in window and points on both sides', [5.8, 6.1, 14.1, 8, 12], 50),
            ('second_lowest in window but all above and not close', [5.8, 11.3, 14.1, 11.3, 12], np.nan),
            ('second_lowest in window but all below and not close', [5.8, 6.1, 14.1, 8, 8.7], np.nan),
        ]
        for case, energy, expected_output in test_cases:
            with self.subTest(case):
                config = build_swe_configuration(
                    pitch_angle_bins=[45],
                    pitch_angle_deltas=[45],
                    energy_bins=[10],
                    energy_bin_low_multiplier=0.6,
                    energy_bin_high_multiplier=1.4,
                    high_energy_proximity_threshold=0.18,
                    low_energy_proximity_threshold=0.12
                )

                pitch_angle = np.array([30, 35, 40, 45, 50])
                psd = np.array([50, 50, 50, 50, 50])
                energy = np.array(energy)

                result = rebin_by_pitch_angle(psd, pitch_angle, energy, config)

                np.testing.assert_almost_equal(result, np.array([[expected_output]]))

    def test_rebin_full_size_data(self):
        rng = np.random.default_rng(202502201113)
        flux = rng.random((24, 30, 7)) * 1000
        pitch_angle = rng.random((24, 30, 7)) * 180
        energy = rng.random((24, 30, 7)) * 1000
        pitch_angle_bins = np.linspace(0, 180, 20, endpoint=False) + 4.5
        pitch_angle_deltas = np.repeat(4.5, 20)
        energy_bins = np.geomspace(2, 5000, 24)
        config = build_swe_configuration(pitch_angle_bins=pitch_angle_bins,
                                         pitch_angle_deltas=pitch_angle_deltas,
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

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.rebin_by_pitch_angle_and_gyrophase')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_gyrophase')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_energy_in_ev_from_velocity_in_km_per_second')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.rebin_by_pitch_angle')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_pitch_angle')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_velocity_in_sw_frame')
    def test_correct_and_rebin(self, mock_calculate_velocity_in_sw_frame,
                               mock_calculate_pitch_angle, mock_rebin_by_pitch_angle, mock_calculate_energy,
                               mock_calculate_gyrophase, mock_rebin_by_pitch_angle_and_gyrophase):

        mag_vectors = np.array([
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 1],
                [1, 1, 1],
            ],
        ])

        flux_data = Mock()
        solar_wind_vector = Mock()
        configuration = Mock()
        result = correct_and_rebin(
            flux_or_psd=flux_data,
            solar_wind_vector=solar_wind_vector,
            dsp_velocities=sentinel.dsp_velocities,
            mag_vector=mag_vectors,
            config=configuration,
        )

        expected_mag_vectors_with_cem_axis = np.array([
            [
                [[1, 0, 0]],
                [[0, 1, 0]],
            ],
            [
                [[0, 0, 1]],
                [[1, 1, 1]],
            ],
        ])

        mock_calculate_velocity_in_sw_frame.assert_called_once_with(sentinel.dsp_velocities, solar_wind_vector)

        mock_calculate_pitch_angle.assert_called_once_with(mock_calculate_velocity_in_sw_frame.return_value,
                                                           NumpyArrayMatcher(expected_mag_vectors_with_cem_axis))
        mock_calculate_gyrophase.assert_called_once_with(mock_calculate_velocity_in_sw_frame.return_value,
                                                         NumpyArrayMatcher(expected_mag_vectors_with_cem_axis))

        mock_calculate_energy.assert_called_with(mock_calculate_velocity_in_sw_frame.return_value)
        mock_rebin_by_pitch_angle.assert_called_with(
            flux_data, mock_calculate_pitch_angle.return_value,
            mock_calculate_energy.return_value,
            configuration,
        )
        mock_rebin_by_pitch_angle_and_gyrophase.assert_called_with(
            flux_data,
            mock_calculate_pitch_angle.return_value,
            mock_calculate_gyrophase.return_value,
            mock_calculate_energy.return_value,
            configuration,
        )
        self.assertEqual((mock_rebin_by_pitch_angle.return_value, mock_rebin_by_pitch_angle_and_gyrophase.return_value),
                         result)

    def test_integrate_distribution_to_get_1d_spectrum(self):
        pitch_angle_bins = [30, 90]
        pitch_angle_deltas = [15, 15]
        configuration = build_swe_configuration(pitch_angle_bins=pitch_angle_bins,
                                                pitch_angle_deltas=pitch_angle_deltas,
                                                in_vs_out_energy_index=1,
                                                )

        psd_by_pitch_angles = np.array([
            [10, 20],
            [5, 10],
            [7, 15],
        ])

        actual_integrated_spectrum = integrate_distribution_to_get_1d_spectrum(psd_by_pitch_angles, configuration)

        bin1_factor = ((0.5 * np.deg2rad(30)) / 2)
        bin2_factor = ((1 * np.deg2rad(30)) / 2)
        expected_integrated_spectrum = [
            10 * bin1_factor + 20 * bin2_factor,
            5 * bin1_factor + 10 * bin2_factor,
            7 * bin1_factor + 15 * bin2_factor
        ]
        np.testing.assert_allclose(actual_integrated_spectrum, expected_integrated_spectrum)

    def test_integrate_distribution_to_get_1d_spectrum_ignores_nan_values(self):
        pitch_angle_bins = [30, 90]
        pitch_angle_deltas = [15, 15]
        configuration = build_swe_configuration(pitch_angle_bins=pitch_angle_bins,
                                                pitch_angle_deltas=pitch_angle_deltas,
                                                in_vs_out_energy_index=1,
                                                )

        psd_by_energy_and_pitch_angles = np.array([
            [10, np.nan],
            [5, 10],
            [np.nan, 15],
        ])

        actual_integrated_spectrum = integrate_distribution_to_get_1d_spectrum(psd_by_energy_and_pitch_angles,
                                                                               configuration)

        bin1_factor = ((0.5 * np.deg2rad(30)) / 2)
        bin2_factor = ((1 * np.deg2rad(30)) / 2)
        expected_integrated_spectrum = [
            10 * bin1_factor,
            5 * bin1_factor + 10 * bin2_factor,
            15 * bin2_factor
        ]
        np.testing.assert_allclose(actual_integrated_spectrum, expected_integrated_spectrum)

    def test_integrate_distribution_decides_inbound_and_outbound_based_on_config_energy_index(
            self):
        pitch_angle_bins = [10, 85.5, 94.5, 100]
        pitch_angle_deltas = [4.5, 4.5, 4.5, 4.5]

        psd_by_pitch_angles = np.array([
            [10, 20, 30, 40],
            [15, 20, 5, 10],
            [1, 2, 3, 4],
        ])

        bin1_factor = (np.sin(np.deg2rad(10)) * np.deg2rad(4.5 * 2))
        bin2_factor = (np.sin(np.deg2rad(85.5)) * np.deg2rad(4.5 * 2))
        bin3_factor = (np.sin(np.deg2rad(94.5)) * np.deg2rad(4.5 * 2))
        bin4_factor = (np.sin(np.deg2rad(100)) * np.deg2rad(4.5 * 2))

        expected_A_spectrum = [
            10 * bin1_factor + 20 * bin2_factor,
            15 * bin1_factor + 20 * bin2_factor,
            1 * bin1_factor + 2 * bin2_factor,
        ]
        expected_B_spectrum = [
            30 * bin3_factor + 40 * bin4_factor,
            5 * bin3_factor + 10 * bin4_factor,
            3 * bin3_factor + 4 * bin4_factor,
        ]

        cases = [
            (1, expected_B_spectrum, expected_A_spectrum),
            (2, expected_A_spectrum, expected_B_spectrum),
        ]
        for index, expected_in, expected_out in cases:
            with self.subTest(index):
                configuration = build_swe_configuration(pitch_angle_bins=pitch_angle_bins,
                                                        pitch_angle_deltas=pitch_angle_deltas,
                                                        in_vs_out_energy_index=index)

                in_spectrum, out_spectrum = integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(
                    psd_by_pitch_angles,
                    configuration)

                np.testing.assert_allclose(in_spectrum, expected_in)
                np.testing.assert_allclose(out_spectrum, expected_out)

    def test_integrate_distribution_to_get_inbound_and_outbound_ignores_fill(
            self):
        pitch_angle_bins = [10, 85.5, 94.5, 100]
        pitch_angle_deltas = [4.5, 4.5, 4.5, 4.5]

        psd_by_pitch_angles = np.array([
            [10, 20, 30, 40],
            [np.nan, 20, 5, 10],
            [1, 2, 3, np.nan],
        ])

        bin1_factor = (np.sin(np.deg2rad(10)) * np.deg2rad(4.5 * 2))
        bin2_factor = (np.sin(np.deg2rad(85.5)) * np.deg2rad(4.5 * 2))
        bin3_factor = (np.sin(np.deg2rad(94.5)) * np.deg2rad(4.5 * 2))
        bin4_factor = (np.sin(np.deg2rad(100)) * np.deg2rad(4.5 * 2))

        expected_A_spectrum = [
            10 * bin1_factor + 20 * bin2_factor,
            0 + 20 * bin2_factor,
            1 * bin1_factor + 2 * bin2_factor,
        ]
        expected_B_spectrum = [
            30 * bin3_factor + 40 * bin4_factor,
            5 * bin3_factor + 10 * bin4_factor,
            3 * bin3_factor + 0,
        ]

        configuration = build_swe_configuration(pitch_angle_bins=pitch_angle_bins,
                                                pitch_angle_deltas=pitch_angle_deltas,
                                                in_vs_out_energy_index=0)

        in_spectrum, out_spectrum = integrate_distribution_to_get_inbound_and_outbound_1d_spectrum(
            psd_by_pitch_angles,
            configuration)

        np.testing.assert_allclose(in_spectrum, expected_A_spectrum)
        np.testing.assert_allclose(out_spectrum, expected_B_spectrum)

    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_unit_vector')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_pitch_angle')
    @patch('imap_l3_processing.swe.l3.science.pitch_calculations.calculate_gyrophase')
    def test_rebin_swe_intensity(self, mock_calculate_gyrophases, mock_calculate_pitch_angles,
                                 mock_calculate_unit_vector):
        intensity_data_for_energy_1 = [[0, 1, 2, 3], [4, 5, 6, 7]]
        intensity_data_for_energy_2 = [[8, 9, 10, 11], [12, 13, 14, 15]]
        intensity = np.array([
            intensity_data_for_energy_1,
            intensity_data_for_energy_2
        ])

        pitch_angles = np.array(
            [
                [[45, 45, 135, 135], [45, 45, 135, 135]],
                [[45, 45, 135, 135], [45, 45, 135, 135]],
            ])
        mock_calculate_pitch_angles.return_value = pitch_angles

        gyrophases = np.array(
            [
                [[135, 225, 225, 135], [45, 315, 315, 45]],
                [[135, 225, 225, 135], [45, 315, 315, 45]]
            ])
        mock_calculate_gyrophases.return_value = gyrophases

        counts = np.array([
            [[4, 9, 49, 36], [0, 16, 64, 25]],
            [[4 * 4, 4 * 9, 4 * 49, 4 * 36], [4 * 1, 4 * 16, 4 * 64, 4 * 25]],
        ])

        normalized_mag_vectors = np.array([
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 1],
                [1, 1, 1],
            ],
        ])
        expected_mag_vectors_with_cem_axis = np.array([
            [
                [[1, 0, 0]],
                [[0, 1, 0]],
            ],
            [
                [[0, 0, 1]],
                [[1, 1, 1]],
            ],
        ])
        mock_calculate_unit_vector.side_effect = [sentinel.normalized_dsp_velocities, normalized_mag_vectors]

        config = build_swe_configuration(
            pitch_angle_bins=np.array([45, 135]),
            pitch_angle_deltas=np.array([45, 45]),
            gyrophase_bins=np.array([45, 135, 225, 315]),
            gyrophase_deltas=np.array([45, 45, 45, 45]),
        )
        expected_intensity_by_pa_and_gyro = np.array([
            [[4, 0, 1, 5], [7, 3, 2, 6]],
            [[12, 8, 9, 13], [15, 11, 10, 14]]
        ])
        expected_intensity_by_pa = np.array([[2.5, 4.5], [10.5, 12.5]])

        expected_uncertainty_by_pa_gyro = expected_intensity_by_pa_and_gyro * np.array([
            [[np.nan, 1 / 2, 1 / 3, 1 / 4], [1 / 5, 1 / 6, 1 / 7, 1 / 8]],
            [[1 / 2, 1 / 4, 1 / 6, 1 / 8], [1 / 10, 1 / 12, 1 / 14, 1 / 16]]
        ])
        expected_uncertainty_by_pa = expected_intensity_by_pa * np.array(
            [[1 / np.sqrt(4 + 9 + 0 + 16), 1 / np.sqrt(49 + 36 + 64 + 25)],
             [1 / (2 * np.sqrt(4 + 9 + 1 + 16)),
              1 / (2 * np.sqrt(49 + 36 + 64 + 25))]])

        rebinned_data = swe_rebin_intensity_by_pitch_angle_and_gyrophase(intensity,
                                                                         counts,
                                                                         sentinel.dsp_velocities,
                                                                         sentinel.mag_vectors, config)
        mock_calculate_unit_vector.assert_has_calls([call(sentinel.dsp_velocities), call(sentinel.mag_vectors)])
        mock_calculate_pitch_angles.assert_called_once_with(sentinel.normalized_dsp_velocities,
                                                            NumpyArrayMatcher(expected_mag_vectors_with_cem_axis))
        mock_calculate_gyrophases.assert_called_once_with(sentinel.normalized_dsp_velocities,
                                                          NumpyArrayMatcher(expected_mag_vectors_with_cem_axis))
        actual_rebinned_by_pa_and_gyro, actual_rebinned_by_pa, actual_intensity_uncertainty_by_pa_and_gyro, actual_intensity_uncertainty_by_pa = rebinned_data
        np.testing.assert_equal(actual_rebinned_by_pa_and_gyro, expected_intensity_by_pa_and_gyro)
        np.testing.assert_equal(actual_rebinned_by_pa, expected_intensity_by_pa)
        np.testing.assert_array_almost_equal(actual_intensity_uncertainty_by_pa_and_gyro,
                                             expected_uncertainty_by_pa_gyro)
        np.testing.assert_array_almost_equal(actual_intensity_uncertainty_by_pa, expected_uncertainty_by_pa)

    def test_lsfit(self):

        initial_guess = [np.float64(4.473391704937145e-25), np.float64(0.19096020638749855),
                         np.float64(6.063340103406818), np.float64(0.12037186161346533), np.float64(83.52704460466526),
                         np.float64(0.040282464591878904)]

        energies = np.array(
            [2.55714286, 3.65142857, 5.16, 7.30571429, 10.32857143, 14.34285714, 19.95714286, 27.42857143, 38.37142857,
             52.82857143, 73.32857143, 102.0, 142.14285714, 196.57142857, 272.0])

        psd = np.array(
            [-56.55479296157507, -56.76375798660356, -57.53496465790231, -58.00961908572603, -58.14254928568969,
             -58.368814265394995, -58.95578128605549, -59.85513105193842, -61.139029802796635, -62.73189593308678,
             -64.31852773524902, -65.47348354141863, -66.51468253177605, -67.62939966827234, -68.79150149983676])

        expected_fit = np.array(
            [6.71476597e-25, 3.42058474e-01, 6.00716325e+00, 1.01128559e-01, 8.01148311e+01, 1.92992352e-02])

        fit = ls_fit(energies, psd, initial_guess)

        np.testing.assert_array_almost_equal(fit, expected_fit)

    def test_try_curve_fit_until_valid_gives_up_after_4_bad_fits(self):
        initial_bad_guess = [5.925797921175077e-25, 0.18724903118073213, 4.112440864015557,
                             0.15397363076239517, 74.23743613506248, 0.029443756322633966]
        energies = np.array(
            [2.55714286, 3.65142857, 5.16, 7.30571429, 10.32857143, 14.34285714, 19.95714286, 27.42857143,
             38.37142857, 52.82857143, 73.32857143, 102.0, 142.14285714, 196.57142857, 272.0, 372.71428571])

        psd = np.array(
            [-56.26413449978238, -56.4690384388148, -56.77314225747658, -56.988215330802944, -57.25856400334476,
             -57.5797875237785, -58.318166428419275, -59.468569412324065, -61.17996364351136, -63.27412512536183,
             -64.63723959633855, -65.48143415257543, -66.35949432020823, -67.38274526815847, -68.50579882255904, -
             69.64891974378229])

        spacecraft_potential, halo_breakpoint = try_curve_fit_until_valid(energies, psd, initial_bad_guess, 3, 100,
                                                                          -1.5, -10)
        self.assertEqual(spacecraft_potential, 3)
        self.assertEqual(halo_breakpoint, 100)

    def test_try_curve_fit_until_valid_attempts_again(self):

        initial_guess = [np.float64(3.1393252613729137e-24), np.float64(0.43616983124497205),
                         np.float64(14.194616719775487), np.float64(0.04172457075833008), np.float64(91.99702053644349),
                         np.float64(0.040856710760706225)]

        energies = np.array(
            [2.55714286, 3.65142857, 5.16, 7.30571429, 10.32857143, 14.34285714, 19.95714286, 27.42857143, 38.37142857,
             52.82857143, 73.32857143, 102.0, 142.14285714, 196.57142857, 272.0, 372.71428571])

        psd = np.array(
            [-55.23338290965658, -55.71067732312106, -56.318756521655594, -56.82755277520815, -57.23748711228458,
             -58.78220038027476, -59.01645404205638, -59.45473432132905, -60.235106156756885, -61.354895453684456,
             -62.71973274071986, -63.8911530049006, -64.9940700713914, -66.19502869267995, -67.43704033083272, -
             68.78958034891963])

        latest_spacecraft_potential = 14.129395820280818
        latest_core_halo_breakpoint = 94.53406293103626
        spacecraft_potential, halo_breakpoint = try_curve_fit_until_valid(energies, psd, initial_guess,
                                                                          latest_spacecraft_potential,
                                                                          latest_core_halo_breakpoint, -1.5, -10)

        self.assertAlmostEqual(spacecraft_potential, 14.874387243851425)
        self.assertAlmostEqual(halo_breakpoint, 109.85461635815749)

    def test_spacecraft_potential_matches_heritage_code(self):
        energies = np.array([2.55714286, 3.65142857, 5.16, 7.30571429, 10.32857143, 14.34285714, 19.95714286,
                             27.42857143, 38.37142857, 52.82857143, 73.32857143, 102.0, 142.14285714,
                             196.57142857, 272.0, 372.71428571, 519.0, 712.57142857, 987.14285714, 1370.0])

        cases = [
            (
                [3.23283052e-25, 2.55480962e-25, 1.26704353e-25, 7.55743191e-26,
                 6.16583651e-26, 5.01600338e-26, 2.86150808e-26, 1.18947338e-26,
                 3.29208997e-27, 6.79583049e-28, 1.46010436e-28, 4.61815072e-29,
                 1.64616022e-29, 5.33855137e-30, 1.62662108e-30, 4.93599290e-31,
                 1.41154820e-31, 4.10384322e-32, 1.49581193e-32, 1.00498754e-32],
                [10],
                [80],
                6.37612093296215,
                82.6628354261301,
            ),
            (
                [3.813903545692864e-25, 3.006442020782414e-25,
                 1.3129686941831112e-25, 9.668812518532953e-26, 7.15914007349482e-26,
                 5.730862490659475e-26, 3.235795236983263e-26, 1.295168689217677e-26,
                 3.435681475566846e-27, 7.548450205282862e-28, 2.0195915946797056e-28,
                 7.29483542198608e-29, 2.7293948869964095e-29, 8.84074868812542e-30,
                 2.636133339901512e-30, 7.457543785597129e-31, 1.7351203177626105e-31,
                 3.7391585374572035e-32, 1.1503823740807066e-32, 1.0000418721498402e-32],
                [5.429858684539795, 5.5335798263549805, 5.422885417938232],
                [80.37294006347656, 80.40635681152344, 80.59530639648438],
                5.30492403628744,
                66.7688700186013,
            )
        ]

        config = build_swe_configuration()
        for index, (psd_list, latest_potentials, latest_breaks, expected_potential, expected_break) in enumerate(cases):
            with self.subTest(index):
                psd = np.array(psd_list)

                potential, core_halo_break = find_breakpoints(energies, psd, latest_potentials, latest_breaks, config)
                np.testing.assert_allclose(expected_potential, potential, rtol=1e-12)
                np.testing.assert_allclose(expected_break, core_halo_break, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
