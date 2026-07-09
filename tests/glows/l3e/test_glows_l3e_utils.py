import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, call, sentinel

import numpy as np
from imap_data_access.file_validation import Version
from imap_processing.spice.repoint import get_repoint_data, set_global_repoint_table_paths
from spacepy.pycdf import CDF, const

from imap_l3_processing.glows.descriptors import GLOWS_L3E_DESCRIPTORS, GLOWS_L3E_HI_45_DESCRIPTOR, \
    GLOWS_L3E_HI_90_DESCRIPTOR, GLOWS_L3E_LO_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR, GLOWS_L3E_ULTRA_HF_DESCRIPTOR
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable, \
    identify_versions_for_l3e_output_files, find_first_updated_cr, get_lo_pivot_angles, \
    get_lo_pivot_angle_from_l1b_file, LoPivotAngle, compute_glows_flags_for_window, \
    get_repoint_numbers_within_cr_window
from imap_l3_processing.models import VersionMap
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
            'glows_flags': np.arange(num_crs),
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

        new_glows_flags = np.arange(num_crs)
        new_glows_flags[9] = 10

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
            ('glows_flags', new_glows_flags, 9),
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

                actual_cr = find_first_updated_cr(
                    sentinel.new_l3d_path, sentinel.old_l3d_filename
                )

                mock_download.assert_called_once_with(sentinel.old_l3d_filename)
                mock_CDF.assert_has_calls(
                    [
                        call(str(sentinel.download_old_l3d)),
                        call(str(sentinel.new_l3d_path)),
                    ]
                )

                self.assertEqual(actual_cr, expected)

    def test_get_lo_pivot_angle_from_l1b_file_real_cdf(self):
        l1b_file = get_test_data_path(
            "glows/imap_lo_l1b_nhk_20260318-repoint00189_v003.cdf"
        )
        actual = get_lo_pivot_angle_from_l1b_file(l1b_file)
        self.assertEqual(90.0, actual)

    def test_get_lo_pivot_angle_from_l1b_file_scenarios(self):
        first_thirty_minutes = [
            datetime(2026, 3, 20, 0, 15),
            datetime(2026, 3, 20, 0, 30),
        ]
        thirty_minutes_to_22_hours_thirty_minutes = [
            datetime(2026, 3, 20, 4, 45),
            datetime(2026, 3, 20, 8, 45),
            datetime(2026, 3, 20, 10, 45),
            datetime(2026, 3, 20, 12, 45),
            datetime(2026, 3, 20, 14, 45),
        ]
        after_22_hours_thirty_minutes = [
            datetime(2026, 3, 20, 23, 0),
            datetime(2026, 3, 20, 23, 30),
        ]
        epochs = first_thirty_minutes + thirty_minutes_to_22_hours_thirty_minutes + after_22_hours_thirty_minutes
        shifted_epochs = [e + timedelta(hours=10) for e in epochs]
        cases = [
            ("realistic", epochs, [89.1, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9, 89.9], 90),
            ("basic", epochs, [10, 20, 30, 40, 50, 60, 70, 80, 90], 50),
            ("only uses data within 3-15 hours from first point", shifted_epochs,
             [999, 999, 30, 40, 50, 60, 70, 999, 999], 50),
            ("uses median and rounds", epochs, [999, 999, 120.2, 34.4, 86.8, 50.9, 77.7, 999, 999], 78),
            ("fallback to 90 if no points in interval", first_thirty_minutes + after_22_hours_thirty_minutes,
             [10, 10, 10, 10], 90),
            ("fallback to 90 if no points at all", [], [], 90),
        ]
        for name, epochs, pivot_angles, expected in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cdf_path = Path(tmp_dir, "l1b.cdf")
                    with CDF(str(cdf_path), create=True) as cdf:
                        cdf["epoch"] = epochs
                        cdf["pcc_coarse_pot_pri"] = pivot_angles
                    actual = get_lo_pivot_angle_from_l1b_file(cdf_path)
                    self.assertEqual(expected, actual)

    def test_compute_glows_flags_for_window(self):
        epochs = [
            datetime(2025, 5, 1, 0, 0),
            datetime(2025, 5, 1, 12, 0),
            datetime(2025, 5, 2, 0, 0),
            datetime(2025, 5, 2, 12, 0),
            datetime(2025, 5, 3, 0, 0),
        ]
        flags = [1, 4, 8, 16, 2]

        cases = [
            ("ORs multiple rows inside window", datetime(2025, 5, 1, 6, 0), datetime(2025, 5, 2, 18, 0), 28),
            ("excludes rows before window", datetime(2025, 5, 1, 6, 0), datetime(2025, 5, 1, 18, 0), 4),
            ("excludes rows after window", datetime(2025, 5, 1, 18, 0), datetime(2025, 5, 2, 6, 0), 8),
            ("includes boundaries inclusively", datetime(2025, 5, 1, 0, 0), datetime(2025, 5, 3, 0, 0), 31),
            ("returns zero when no rows in window", datetime(2025, 5, 5, 0, 0), datetime(2025, 5, 6, 0, 0), 0),
        ]

        for name, window_start, window_end, expected in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cdf_path = Path(tmp_dir, "l3d.cdf")
                    with CDF(str(cdf_path), create=True) as cdf:
                        cdf["epoch"] = epochs
                        cdf.new("glows_flags", data=flags, type=const.CDF_UINT2, recVary=True)

                    actual = compute_glows_flags_for_window(cdf_path, window_start, window_end)

                    self.assertIsInstance(actual, int)
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

    def test_get_repoint_numbers_within_cr_window(self):
        start_cr = 2093
        end_cr = 2094
        expected_repoint_numbers = list(range(3682, 3736))

        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")

        set_global_repoint_table_paths([repointing_path])
        repointing_data = get_repoint_data()

        actual_repoint_numbers = get_repoint_numbers_within_cr_window(start_cr, end_cr, repointing_data)

        np.testing.assert_array_equal(actual_repoint_numbers, expected_repoint_numbers)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_gives_minor_version_1_for_non_existing_l3e(self,
                                                                                               mock_imap_data_access,
                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([])
        ]

        all_repointing_numbers = list(range(3682, 3736))
        updated_repointing_numbers = list()
        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission, first_cr_updated_in_l3d,
                                                        repointing_path, version_map)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(3, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(4, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(5, 1) for repoint_number in
                                                   all_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(6, 1) for repoint_number in
                                                         all_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(7, 1) for repoint_number in
                                                         all_repointing_numbers}

        self.assertCountEqual(all_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings)
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)


    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_gives_minor_version_1_for_non_existing_l3e(self,
                                                                                               mock_imap_data_access,
                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({}, Version(None, 1))

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([]),
            create_mock_query_results([])
        ]

        all_repointing_numbers = list(range(3682, 3736))
        updated_repointing_numbers = list()
        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission, first_cr_updated_in_l3d,
                                                        repointing_path, version_map)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(None, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(None, 1) for repoint_number in
                                                     all_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(None, 1) for repoint_number in
                                                   all_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(None, 1) for repoint_number in
                                                         all_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(None, 1) for repoint_number in
                                                         all_repointing_numbers}

        self.assertCountEqual(all_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings)
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_increments_major_and_minor_when_given_higher_major_version(self,
                                                                                                               mock_imap_data_access,
                                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        all_repointing_numbers = list(range(3682, 3736))
        updated_repointing_numbers = list()

        cases = (2, None)
        for old_major_version in cases:
            with self.subTest(old_major_version):
                mock_get_repoint_numbers_within_cr_window.reset_mock()
                mock_imap_data_access.reset_mock()

                mock_get_repoint_numbers_within_cr_window.side_effect = [
                    all_repointing_numbers,
                    updated_repointing_numbers
                ]

                mock_imap_data_access.query.side_effect = [
                    create_mock_query_results([
                        f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03682_{Version(old_major_version, 1)}.cdf',
                        f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03683_{Version(3, 1)}.cdf',
                        f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint03735_{Version(3, 1)}.cdf'
                    ]),
                    create_mock_query_results([
                        f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03683_{Version(old_major_version, 2)}.cdf',
                        f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03684_{Version(4, 2)}.cdf',
                        f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint03735_{Version(4, 2)}.cdf'
                    ]),
                    create_mock_query_results([
                        f'imap_glows_l3e_survival-probability-lo_20250101-repoint03684_{Version(old_major_version, 3)}.cdf',
                        f'imap_glows_l3e_survival-probability-lo_20250101-repoint03685_{Version(5, 3)}.cdf',
                        f'imap_glows_l3e_survival-probability-lo_20250101-repoint03735_{Version(5, 3)}.cdf'
                    ]),
                    create_mock_query_results([
                        f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03685_{Version(old_major_version, 4)}.cdf',
                        f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03686_{Version(6, 4)}.cdf',
                        f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint03735_{Version(6, 4)}.cdf'
                    ]),
                    create_mock_query_results([
                        f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03686_{Version(old_major_version, 5)}.cdf',
                        f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03687_{Version(7, 5)}.cdf',
                        f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint03735_{Version(7, 5)}.cdf'
                    ])
                ]

                result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                                first_cr_updated_in_l3d, repointing_path, version_map)

                mock_imap_data_access.query.assert_has_calls([
                    call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
                    call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
                    call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
                    call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
                    call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
                ])

                self.assertCountEqual(list(range(3682, 3735)), result.repointing_numbers)

                self.assertNotIn(3683, result.hi_45_repointings)
                self.assertNotIn(3684, result.hi_90_repointings)
                self.assertNotIn(3685, result.lo_repointings)
                self.assertNotIn(3686, result.ultra_sf_repointings)
                self.assertNotIn(3687, result.ultra_hf_repointings)

                self.assertEqual(Version(3, 2), result.hi_45_repointings[3682])
                self.assertEqual(Version(4, 3), result.hi_90_repointings[3683])
                self.assertEqual(Version(5, 4), result.lo_repointings[3684])
                self.assertEqual(Version(6, 5), result.ultra_sf_repointings[3685])
                self.assertEqual(Version(7, 6), result.ultra_hf_repointings[3686])

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_does_not_process_if_input_major_version_is_none_and_existing_has_major(self,
                                                                                                               mock_imap_data_access,
                                                                                                               mock_get_repoint_numbers_within_cr_window):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2094
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({}, Version(None,1))

        all_repointing_numbers = list(range(3682, 3736))
        updated_repointing_numbers = list()

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint{repoint:05d}_v001.0001.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint{repoint:05d}_v001.0002.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint{repoint:05d}_v001.0003.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint{repoint:05d}_v001.0004.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint{repoint:05d}_v001.0005.cdf' for repoint in all_repointing_numbers
            ])
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                        first_cr_updated_in_l3d, repointing_path, version_map)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        self.assertEqual(0, len(result.repointing_numbers))
        self.assertEqual({}, result.hi_45_repointings)
        self.assertEqual({}, result.hi_90_repointings)
        self.assertEqual({}, result.lo_repointings)
        self.assertEqual({}, result.ultra_sf_repointings)
        self.assertEqual({}, result.ultra_hf_repointings)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_increments_minor_when_same_major_and_updated_l3d_covers_pointing(
            self, mock_imap_data_access, mock_get_repoint_numbers_within_cr_window
    ):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2095
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({desc: Version(3 + i, 5) for i, desc in enumerate(GLOWS_L3E_DESCRIPTORS)})

        all_repointing_numbers = list(range(3682, 3763))
        updated_repointing_numbers = list(range(3709, 3763))

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint{repoint:05d}_{Version(3, 1)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint{repoint:05d}_{Version(4, 2)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint{repoint:05d}_{Version(5, 3)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint{repoint:05d}_{Version(6, 4)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint{repoint:05d}_{Version(7, 5)}.cdf' for repoint in all_repointing_numbers
            ])
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                        first_cr_updated_in_l3d, repointing_path, version_map)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(3, 2) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(4, 3) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(5, 4) for repoint_number in
                                                   updated_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(6, 5) for repoint_number in
                                                         updated_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(7, 6) for repoint_number in
                                                         updated_repointing_numbers}

        self.assertCountEqual(updated_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings)
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.get_repoint_numbers_within_cr_window')
    @patch('imap_l3_processing.glows.l3e.glows_l3e_utils.imap_data_access')
    def test_identify_versions_for_l3e_output_files_increments_minor_for_legacy_versioning_and_updated_l3d_covers_pointing(
            self, mock_imap_data_access, mock_get_repoint_numbers_within_cr_window
    ):
        start_cr_of_mission = 2093
        end_cr_of_mission = 2095
        first_cr_updated_in_l3d = None
        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")
        version_map = VersionMap({}, Version(None,1))

        all_repointing_numbers = list(range(3682, 3763))
        updated_repointing_numbers = list(range(3709, 3763))

        mock_get_repoint_numbers_within_cr_window.side_effect = [
            all_repointing_numbers,
            updated_repointing_numbers
        ]

        mock_imap_data_access.query.side_effect = [
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-90_20250101-repoint{repoint:05d}_{Version(None, 1)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-hi-45_20250101-repoint{repoint:05d}_{Version(None, 2)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-lo_20250101-repoint{repoint:05d}_{Version(None, 3)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-sf_20250101-repoint{repoint:05d}_{Version(None, 4)}.cdf' for repoint in all_repointing_numbers
            ]),
            create_mock_query_results([
                f'imap_glows_l3e_survival-probability-ul-hf_20250101-repoint{repoint:05d}_{Version(None, 5)}.cdf' for repoint in all_repointing_numbers
            ])
        ]

        result = identify_versions_for_l3e_output_files(start_cr_of_mission, end_cr_of_mission,
                                                        first_cr_updated_in_l3d, repointing_path, version_map)

        mock_imap_data_access.query.assert_has_calls([
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_45_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_HI_90_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_LO_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR),
            call(instrument='glows', data_level='l3e', version="latest", descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR)
        ])

        expected_versions_for_hi45_repoint_number = {repoint_number: Version(None, 2) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_hi90_repoint_number = {repoint_number: Version(None, 3) for repoint_number in
                                                     updated_repointing_numbers}
        expected_versions_for_lo_repoint_number = {repoint_number: Version(None, 4) for repoint_number in
                                                   updated_repointing_numbers}
        expected_versions_for_ultra_sf_repoint_number = {repoint_number: Version(None, 5) for repoint_number in
                                                         updated_repointing_numbers}
        expected_versions_for_ultra_hf_repoint_number = {repoint_number: Version(None, 6) for repoint_number in
                                                         updated_repointing_numbers}

        self.assertCountEqual(updated_repointing_numbers, result.repointing_numbers)
        self.assertEqual(expected_versions_for_hi90_repoint_number, result.hi_90_repointings)
        self.assertEqual(expected_versions_for_hi45_repoint_number, result.hi_45_repointings)
        self.assertEqual(expected_versions_for_lo_repoint_number, result.lo_repointings)
        self.assertEqual(expected_versions_for_ultra_sf_repoint_number, result.ultra_sf_repointings)
        self.assertEqual(expected_versions_for_ultra_hf_repoint_number, result.ultra_hf_repointings)