import unittest
from datetime import datetime
from unittest.mock import patch, Mock, call

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable, \
    determine_l3e_files_to_produce, find_first_updated_cr, find_l3e_spice_kernels_for_time_range
from tests.test_helpers import get_test_data_path, create_glows_mock_query_results


class TestGlowsL3EUtils(unittest.TestCase):

    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.datetime2et")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.spkezr")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.reclat")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.pxform")
    def test_determine_call_args_for_l3e_executable(self, mock_pxform: Mock, mock_reclat: Mock, mock_spkezr: Mock,
                                                    mock_date_time_2et: Mock):
        start_time = datetime.fromisoformat("2025-05-01 00:00:00")
        repointing_midpoint = datetime.fromisoformat("2025-05-01 12:00:00")

        x, y, z, vx, vy, vz = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        position_data = [x, y, z, vx, vy, vz]
        mock_spkezr.return_value = (position_data, Mock())

        radius, longitude, latitude = 70000000, -8.0, -.9

        rotation_matrix = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
        mock_pxform.return_value = rotation_matrix

        spin_axis_long, spin_axis_lat = -1.4, 0.2
        mock_reclat.side_effect = [(radius, longitude, latitude), (Mock(), spin_axis_long, spin_axis_lat)]

        elongation = 90
        call_args: GlowsL3eCallArguments = determine_call_args_for_l3e_executable(start_time, repointing_midpoint,
                                                                                  elongation)

        mock_date_time_2et.assert_called_once_with(repointing_midpoint)

        mock_spkezr.assert_called_once_with("IMAP", mock_date_time_2et.return_value, "ECLIPJ2000", "NONE", "SUN")

        np.testing.assert_array_equal([x, y, z], mock_reclat.call_args_list[0][0][0])

        mock_pxform.assert_called_once_with("IMAP_DPS", "ECLIPJ2000", mock_date_time_2et.return_value)
        np.testing.assert_array_equal([12.0, 15.0, 18.0], mock_reclat.call_args_list[1][0][0])

        self.assertEqual("20250501_000000", call_args.formatted_date)
        self.assertEqual("2025.33014", call_args.decimal_date)
        self.assertEqual(0.4679210985587912, call_args.spacecraft_radius)
        self.assertEqual(261.6337638953414, call_args.spacecraft_longitude)
        self.assertEqual(-51.56620156177409, call_args.spacecraft_latitude)
        self.assertEqual(vx, call_args.spacecraft_velocity_x)
        self.assertEqual(vy, call_args.spacecraft_velocity_y)
        self.assertEqual(vz, call_args.spacecraft_velocity_z)
        self.assertEqual(np.rad2deg(spin_axis_long) % 360, call_args.spin_axis_longitude)
        self.assertEqual(np.rad2deg(spin_axis_lat), call_args.spin_axis_latitude)
        self.assertEqual(elongation, call_args.elongation)

    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access.query")
    def test_determine_l3e_files_to_produce(self, mock_query):
        last_processed_cr = 2094
        first_cr_processed = 2093
        repoint_pathing = get_test_data_path("fake_1_day_repointing_file.csv")

        expected_repointings = [i for i in range(3682, 3736)]

        expected_hi_90_repointing_to_version = {i: 1 for i in expected_repointings}
        expected_hi_45_repointing_to_version = {i: 1 for i in expected_repointings}
        expected_lo_repointing_to_version = {i: 1 for i in expected_repointings}
        expected_ultra_repointing_to_version = {i: 1 for i in expected_repointings}

        expected_hi_90_repointing_to_version.update({
            3682: 2, 3683: 3, 3684: 4, 3685: 5, 3686: 6
        })
        expected_hi_45_repointing_to_version.update({
            3683: 3, 3684: 4, 3685: 5, 3686: 6, 3687: 7
        })
        expected_lo_repointing_to_version.update({
            3684: 4, 3685: 5, 3686: 6, 3687: 7, 3688: 8
        })
        expected_ultra_repointing_to_version.update({
            3685: 5, 3686: 6, 3687: 7, 3688: 8, 3689: 9
        })

        mock_query.side_effect = [
            create_glows_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03682_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03683_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03686_v005.cdf',
            ]),
            create_glows_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03683_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03686_v005.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03687_v006.cdf',
            ]),
            create_glows_mock_query_results([
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03686_v005.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03687_v006.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03688_v007.cdf',
            ]),
            create_glows_mock_query_results([
                'imap_glows_l3e_survival-probability-ultra_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-ultra_20250101-repoint03686_v005.cdf',
                'imap_glows_l3e_survival-probability-ultra_20250101-repoint03687_v006.cdf',
                'imap_glows_l3e_survival-probability-ultra_20250101-repoint03688_v007.cdf',
                'imap_glows_l3e_survival-probability-ultra_20250101-repoint03689_v008.cdf',
            ]),
        ]

        repointings = determine_l3e_files_to_produce(first_cr_processed, last_processed_cr, repoint_pathing)
        mock_query.assert_has_calls([
            call(instrument="glows", data_level="l3e", version='latest', descriptor='survival-probability-hi-90'),
            call(instrument="glows", data_level="l3e", version='latest', descriptor='survival-probability-hi-45'),
            call(instrument="glows", data_level="l3e", version='latest', descriptor='survival-probability-lo'),
            call(instrument="glows", data_level="l3e", version='latest', descriptor='survival-probability-ultra'),
        ])

        self.assertEqual(expected_hi_90_repointing_to_version, repointings.hi_90_repointings)
        self.assertEqual(expected_hi_45_repointing_to_version, repointings.hi_45_repointings)
        self.assertEqual(expected_lo_repointing_to_version, repointings.lo_repointings)
        self.assertEqual(expected_ultra_repointing_to_version, repointings.ultra_repointings)
        self.assertEqual(expected_repointings, repointings.repointing_numbers)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.CDF')
    def test_find_first_updated_cr(self, mock_CDF):
        num_crs = 10
        old_l3d = {
            'cr_grid': np.arange(num_crs) + 0.5,
            'lyman_alpha': np.arange(num_crs),
            'phion': np.arange(num_crs),
            'plasma_speed': np.arange(0,20).reshape((num_crs, 2)),
            'plasma_speed_flag': np.arange(num_crs),
            'proton_density': np.arange(0,20).reshape((num_crs, 2)),
            'proton_density_flag': np.arange(num_crs),
            'uv_anisotropy': np.arange(0,20).reshape((num_crs, 2)),
            'uv_anisotropy_flag': np.arange(num_crs),
        }

        cases =['cr_grid',
                'lyman_alpha',
                'phion',
                'plasma_speed',
                'plasma_speed_flag',
                'proton_density',
                'proton_density_flag',
                'uv_anisotropy',
                'uv_anisotropy_flag',
                'no_change']

        for i, case in enumerate(cases):
            with self.subTest(case=case):
                new_l3d = {
                    'cr_grid': np.arange(num_crs),
                    'lyman_alpha': np.arange(num_crs),
                    'phion': np.arange(num_crs),
                    'plasma_speed': np.arange(0, 20).reshape((num_crs, 2)),
                    'plasma_speed_flag': np.arange(num_crs),
                    'proton_density': np.arange(0, 20).reshape((num_crs, 2)),
                    'proton_density_flag': np.arange(num_crs),
                    'uv_anisotropy': np.arange(0, 20).reshape((num_crs, 2)),
                    'uv_anisotropy_flag': np.arange(num_crs),
                }
                mock_CDF.side_effect = [old_l3d, new_l3d]

                if case == "cr_grid":
                    new_l3d['cr_grid'] = np.append(new_l3d['cr_grid'], 10)
                    self.assertEqual(find_first_updated_cr(new_l3d, old_l3d), 10)
                elif case == "no_change":
                    self.assertIsNone(find_first_updated_cr(new_l3d, old_l3d))
                elif case in ['plasma_speed', 'proton_density', 'uv_anisotropy']:
                    new_l3d[case][i] = np.full_like(new_l3d[case][i], -1)
                    self.assertEqual(i, find_first_updated_cr(new_l3d, old_l3d))
                else:
                    new_l3d[case][i] = i + 1
                    self.assertEqual(i, find_first_updated_cr(new_l3d, old_l3d))

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.requests')
    def test_find_l3e_spice_kernels_for_time_range(self, mock_requests):
        desired_spice_start = datetime(2025, 1, 1)
        desired_spice_end = datetime(2025, 3, 1)

        mock_naif_json = [
            {
                "file_name": "lsk/naif0012.tls",
                "min_date_datetime": "2024-12-01, 00:00:00",
                "max_date_datetime": "2025-05-01, 00:00:00"
            },
        ]

        mock_sclk_json = [
            {
                "file_name": "lsk/imap_sclk_0000.tsc",
                "min_date_datetime": "2024-12-01, 00:00:00",
                "max_date_datetime": "2025-05-01, 00:00:00"
            },
        ]

        mock_dps_json = [
            {
                "file_name": "ck/imap_dps_2024_270_2026_335_01.ah.bc",
                "min_date_datetime": "2024-09-01, 00:00:00",
                "max_date_datetime": "2024-12-01, 00:00:00",
            },
            {
                "file_name": "ck/imap_dps_2024_335_2025_031_01.ah.bc",
                "min_date_datetime": "2024-12-01, 00:00:00",
                "max_date_datetime": "2025-02-01, 00:00:00",
            },
            {
                "file_name": "ck/imap_dps_2025_031_2025_120_01.ah.bc",
                "min_date_datetime": "2025-02-01, 00:00:00",
                "max_date_datetime": "2025-05-01, 00:00:00",
            },
        ]

        mock_imap_frame_json = [
            {
                "file_name": "fk/imap_001.tf",
                "min_date_datetime": "2024-12-01, 00:00:00",
                "max_date_datetime": "2025-05-01, 00:00:00"
            },
        ]

        mock_science_frame_json = [
            {
                "file_name": "fk/imap_science_0001.tf",
                "min_date_datetime": "2024-12-01, 00:00:00",
                "max_date_datetime": "2025-05-01, 00:00:00"
            }
        ]

        mock_naif_response = Mock(json=Mock(return_value=mock_naif_json))
        mock_sclk_response = Mock(json=Mock(return_value=mock_sclk_json))
        mock_dps_response = Mock(json=Mock(return_value=mock_dps_json))
        mock_imap_frame_response = Mock(json=Mock(return_value=mock_imap_frame_json))
        mock_science_frame_response = Mock(json=Mock(return_value=mock_science_frame_json))

        mock_requests.get.side_effect = [
            mock_naif_response,
            mock_sclk_response,
            mock_dps_response,
            mock_imap_frame_response,
            mock_science_frame_response
        ]

        desired_kernels = [
            "leapseconds",
            "spacecraft_clock",
            "pointing_attitude",
            "imap_frames",
            "science_frames",
        ]

        spice_kernels = find_l3e_spice_kernels_for_time_range(desired_spice_start, desired_spice_end, required_kernel_types=desired_kernels)

        mock_requests.get.assert_has_calls([
            call("https://api.dev.imap-mission.com/spice-query?type=leapseconds&start_time=0"),
            call("https://api.dev.imap-mission.com/spice-query?type=spacecraft_clock&start_time=0"),
            call("https://api.dev.imap-mission.com/spice-query?type=pointing_attitude&start_time=0"),
            call("https://api.dev.imap-mission.com/spice-query?type=imap_frames&start_time=0"),
            call("https://api.dev.imap-mission.com/spice-query?type=science_frames&start_time=0")
        ])
        self.assertEqual(["naif0012.tls",
                          "imap_sclk_0000.tsc",
                          "imap_dps_2024_335_2025_031_01.ah.bc",
                          "imap_dps_2025_031_2025_120_01.ah.bc",
                          "imap_001.tf",
                          "imap_science_0001.tf"], spice_kernels)