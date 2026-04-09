import unittest
from unittest.mock import Mock, patch, call, sentinel

import numpy as np

from imap_l3_processing.swe.l3.science.pitch_calculations import mec_breakpoint_finder, \
    average_over_look_directions, calculate_velocity_in_dsp_frame_km_s, calculate_look_directions, rebin_by_pitch_angle, \
    correct_and_rebin, calculate_energy_in_ev_from_velocity_in_km_per_second, integrate_distribution_to_get_1d_spectrum, \
    integrate_distribution_to_get_inbound_and_outbound_1d_spectrum, \
    rebin_by_pitch_angle_and_gyrophase, swe_rebin_intensity_by_pitch_angle_and_gyrophase
from tests.test_helpers import build_swe_configuration, NumpyArrayMatcher

from imap_l3_processing.swe.quality_flags import SweL3Flags

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

    def test_mec_breakpoint_finder(self):
        average_psd = np.asarray([3.86611617e-05, 1.75927552e-05, 6.06703495e-06, 1.47614412e-06,
       2.46270751e-07, 4.97864654e-08, 3.90323281e-08, 4.21037028e-08,
       4.07355835e-08, 3.45166807e-08, 2.52075419e-08, 1.54987172e-08,
       7.84932795e-09, 3.32903783e-09, 1.40908421e-09, 8.53206391e-10,
       7.21564216e-10, 6.31903627e-10, 5.07238502e-10, 3.58827457e-10])
        energies = np.asarray([  2.66      ,   3.37852249,   4.29113316,   5.45025936,
         6.92249015,   8.79240175,  11.16741618,  14.18397245,
        18.01536462,  22.88169719,  29.06252952,  36.91293593,
        46.88390382,  59.54824189,  75.63348661,  96.06369752,
       122.01254227, 154.97072104, 196.83160382, 250.        ])

        sc_pot_to_test, ch_break_to_test, quality_flag_to_test = mec_breakpoint_finder(energies, average_psd)
        expected_values = [np.float64(8.79240175),np.float64(75.63348660999998)]
        self.assertAlmostEqual(expected_values[0], sc_pot_to_test)
        self.assertAlmostEqual(expected_values[1], ch_break_to_test)
        self.assertEqual(SweL3Flags.NONE, quality_flag_to_test)

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


if __name__ == '__main__':
    unittest.main()
