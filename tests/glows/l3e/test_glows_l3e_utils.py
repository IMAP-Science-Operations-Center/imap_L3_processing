import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, call, sentinel

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable, \
    determine_l3e_files_to_produce, find_first_updated_cr, get_lo_pivot_angles, \
    get_lo_pivot_angle_from_l1b_file, LoPivotAngle
from tests.test_helpers import get_test_data_path, create_mock_query_results


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
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03682_v001.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03683_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03686_v005.cdf',
            ]),
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03683_v002.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03686_v005.cdf',
                'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03687_v006.cdf',
            ]),
            create_mock_query_results([
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03684_v003.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03685_v004.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03686_v005.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03687_v006.cdf',
                'imap_glows_l3e_survival-probability-lo_20250101-repoint03688_v007.cdf',
            ]),
            create_mock_query_results([
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
            call(instrument="glows", data_level="l3e", version='latest', descriptor='survival-probability-ul'),
        ])

        self.assertEqual(expected_hi_90_repointing_to_version, repointings.hi_90_repointings)
        self.assertEqual(expected_hi_45_repointing_to_version, repointings.hi_45_repointings)
        self.assertEqual(expected_lo_repointing_to_version, repointings.lo_repointings)
        self.assertEqual(expected_ultra_repointing_to_version, repointings.ultra_repointings)
        self.assertEqual(expected_repointings, repointings.repointing_numbers)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access.download')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.CDF')
    def test_find_first_updated_cr(self, mock_CDF, mock_download):
        num_crs = 10
        mock_download.return_value = sentinel.download_old_l3d

        old_l3d = {
            'cr_grid': np.arange(num_crs) + 0.5,
            'lyman_alpha': np.arange(num_crs),
            'phion': np.arange(num_crs),
            'plasma_speed': np.arange(0, 20).reshape((num_crs, 2)),
            'plasma_speed_flag': np.arange(num_crs),
            'proton_density': np.arange(0, 20).reshape((num_crs, 2)),
            'proton_density_flag': np.arange(num_crs),
            'uv_anisotropy': np.arange(0, 20).reshape((num_crs, 2)),
            'uv_anisotropy_flag': np.arange(num_crs),
        }

        new_lyman_alpha = np.arange(num_crs)
        new_lyman_alpha[1] = 10

        new_phion = np.arange(num_crs)
        new_phion[2] = 10

        new_plasma_speed_flag = np.arange(num_crs)
        new_plasma_speed_flag[3] = 10

        new_proton_density_flag = np.arange(num_crs)
        new_proton_density_flag[4] = 10

        new_uv_anisotropy_flag = np.arange(num_crs)
        new_uv_anisotropy_flag[5] = 10

        new_plasma_speed = np.arange(0, 20).reshape((num_crs, 2))
        new_plasma_speed[6, :] = 10

        new_proton_density = np.arange(0, 20).reshape((num_crs, 2))
        new_proton_density[7, :] = 10

        new_uv_anisotropy = np.arange(0, 20).reshape((num_crs, 2))
        new_uv_anisotropy[8, :] = 10

        cases = [
            ('cr_grid', np.append(old_l3d['cr_grid'], 10.5), 10),
            ('lyman_alpha', new_lyman_alpha, 1),
            ('phion', new_phion, 2),
            ('plasma_speed_flag', new_plasma_speed_flag, 3),
            ('proton_density_flag', new_proton_density_flag, 4),
            ('uv_anisotropy_flag', new_uv_anisotropy_flag, 5),

            ('plasma_speed', new_plasma_speed, 6),
            ('proton_density', new_proton_density, 7),
            ('uv_anisotropy', new_uv_anisotropy, 8),
            ('no_change', None, None)
        ]

        for case, change, expected in cases:
            mock_download.reset_mock()
            mock_CDF.reset_mock()

            with self.subTest(case=case):
                new_l3d = {**old_l3d}
                if case != "no_change":
                    new_l3d[case] = change

                mock_CDF.side_effect = [old_l3d, new_l3d]

                actual_cr = find_first_updated_cr(sentinel.new_l3d_path, sentinel.old_l3d_filename)

                mock_download.assert_called_once_with(sentinel.old_l3d_filename)
                mock_CDF.assert_has_calls([
                    call(str(sentinel.download_old_l3d)),
                    call(str(sentinel.new_l3d_path)),
                ])

                self.assertEqual(actual_cr, expected)

    def test_get_lo_pivot_angle_from_l1b_file_real_cdf(self):
        l1b_file = get_test_data_path("glows/imap_lo_l1b_nhk_20260318-repoint00189_v003.cdf")
        actual = get_lo_pivot_angle_from_l1b_file(l1b_file)
        self.assertEqual(90.0, actual)

    def test_get_lo_pivot_angle_from_l1b_file_scenarios(self):
        first_three_hours = [
            datetime(2026, 3, 20, 0, 45),
            datetime(2026, 3, 20, 2, 45),
        ]
        three_to_fifteen = [
            datetime(2026, 3, 20, 4, 45),
            datetime(2026, 3, 20, 8, 45),
            datetime(2026, 3, 20, 10, 45),
            datetime(2026, 3, 20, 12, 45),
            datetime(2026, 3, 20, 14, 45),
        ]
        after_fifteen = [
            datetime(2026, 3, 20, 16, 45),
            datetime(2026, 3, 20, 20, 45),
        ]
        epochs = first_three_hours + three_to_fifteen + after_fifteen
        shifted_epochs = [e + timedelta(hours=10) for e in epochs]
        cases = [
            ("realistic", epochs, [89.1, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9], 90),
            ("basic", epochs, [10, 20, 30, 40, 50, 60, 70, 80, 90], 50),
            ("only uses data within 3-15 hours from first point", shifted_epochs, [999, 999, 30, 40, 50, 60, 70, 999, 999], 50),
            ("uses median and rounds", epochs, [999, 999, 120.2, 34.4, 86.8, 50.9, 77.7, 999, 999], 78),
            ("fallback to 90 if no points in interval", first_three_hours + after_fifteen, [10, 10, 10, 10], 90),
            ("fallback to 90 if no points at all", [], [], 90),
        ]
        for name, epochs, pivot_angles, expected in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cdf_path = Path(tmp_dir,"l1b.cdf")
                    with CDF(str(cdf_path), create=True) as cdf:
                        cdf["epoch"] = epochs
                        cdf["pcc_coarse_pot_pri"] = pivot_angles
                    actual = get_lo_pivot_angle_from_l1b_file(cdf_path)
                    self.assertEqual(expected, actual)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_lo_pivot_angle_from_l1b_file')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_get_lo_pivot_angles(self, mock_imap_data_access, mock_get_pivot_angle_from_file):
        available_repointings = [1, 2, 3, 4, 5, 6]
        mock_imap_data_access.query.return_value = [
            {'file_path': f'file{i}.cdf', 'repointing': i}
            for i in available_repointings
        ]
        mock_imap_data_access.download.side_effect = lambda name: Path("local/path/to", name)
        pivot_angles_by_file_path = {
            Path("local/path/to/file1.cdf"): 25,
            Path("local/path/to/file2.cdf"): 75,
            Path("local/path/to/file3.cdf"): 105,
            Path("local/path/to/file4.cdf"): 90,
            Path("local/path/to/file5.cdf"): 72,
            Path("local/path/to/file6.cdf"): 84,
        }
        def mock_read_from_cdf(path: Path):
            return pivot_angles_by_file_path[path]
        mock_get_pivot_angle_from_file.side_effect = mock_read_from_cdf

        result = get_lo_pivot_angles([3, 4, 6, 10])

        mock_imap_data_access.query.assert_called_once_with(
            instrument="lo",
            data_level="l1b",
            descriptor="nhk",
            version="latest",
        )
        mock_imap_data_access.download.assert_has_calls([
            call("file3.cdf"),
            call("file4.cdf"),
            call("file6.cdf"),
        ])
        self.assertEqual({
            3: LoPivotAngle(parent_filename="file3.cdf", pivot_angle=105),
            4: LoPivotAngle(parent_filename="file4.cdf", pivot_angle=90),
            6: LoPivotAngle(parent_filename="file6.cdf", pivot_angle=84),
            10: LoPivotAngle(parent_filename=None, pivot_angle=90),
        }, result)
