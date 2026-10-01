import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, call

import imap_data_access
import numpy as np
from imap_data_access import SPICEFilePath
from imap_processing.spice.repoint import get_repoint_data, set_global_repoint_table_paths
from spacepy.pycdf import CDF, const

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import (
    GlowsL3eCallArguments,
    GlowsL3eSpacecraftInfo,
)
from imap_l3_processing.glows.l3e.glows_l3e_utils import (
    determine_call_args_for_l3e_executable,
    get_lo_pivot_angles,
    get_lo_pivot_angle_from_l1b_file,
    LoPivotAngle,
    compute_glows_flags_for_repoint,
    get_repoint_numbers_within_cr_window,
    determine_spacecraft_info_for_l3e_executable,
    determine_spacecraft_info_using_predict_if_needed, calculate_energy_deltas,
)
from imap_l3_processing.glows.quality_flags import GlowsL3Flags
from imap_l3_processing.utils import FurnishMetakernelOutput
from tests.integration.integration_test_helpers import mock_imap_data_access, create_metakernel
from tests.test_helpers import get_test_data_path, get_integration_test_data_path


class TestGlowsL3EUtils(unittest.TestCase):

    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.datetime2et")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.spkezr")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.reclat")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.pxform")
    def test_determine_call_args_for_l3e_executable(self, mock_pxform: Mock, mock_reclat: Mock, mock_spkezr: Mock,
                                                    mock_date_time_2et: Mock):
        start_time = datetime.fromisoformat("2025-05-01 00:00:00")
        repointing_midpoint = datetime.fromisoformat("2025-05-01 12:00:00")

        elongation = 90
        spacecraft_info = Mock()
        call_args: GlowsL3eCallArguments = determine_call_args_for_l3e_executable(start_time, repointing_midpoint,
                                                                                  elongation, spacecraft_info)

        self.assertEqual("20250501_000000", call_args.formatted_date)
        self.assertEqual("2025.33014", call_args.decimal_date)
        self.assertEqual(elongation, call_args.elongation)
        self.assertEqual(spacecraft_info, call_args.spacecraft_info)

    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.datetime2et")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.spkezr")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.reclat")
    @patch("imap_l3_processing.glows.l3e.glows_l3e_utils.spiceypy.pxform")
    def test_determine_spacecraft_info_for_l3e_executable(self, mock_pxform: Mock, mock_reclat: Mock, mock_spkezr: Mock,
                                                    mock_date_time_2et: Mock):
        repointing_midpoint = datetime.fromisoformat("2025-05-01 12:00:00")

        x, y, z, vx, vy, vz = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        position_data = [x, y, z, vx, vy, vz]
        mock_spkezr.return_value = (position_data, Mock())

        radius, longitude, latitude = 70000000, -8.0, -.9

        rotation_matrix = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
        mock_pxform.return_value = rotation_matrix

        spin_axis_long, spin_axis_lat = -1.4, 0.2
        mock_reclat.side_effect = [(radius, longitude, latitude), (Mock(), spin_axis_long, spin_axis_lat)]

        call_args: GlowsL3eSpacecraftInfo = determine_spacecraft_info_for_l3e_executable(repointing_midpoint)

        self.assertIsInstance(call_args, GlowsL3eSpacecraftInfo)

        mock_date_time_2et.assert_called_once_with(repointing_midpoint)

        mock_spkezr.assert_called_once_with("IMAP", mock_date_time_2et.return_value, "ECLIPJ2000", "NONE", "SUN")

        np.testing.assert_array_equal([x, y, z], mock_reclat.call_args_list[0][0][0])

        mock_pxform.assert_called_once_with("IMAP_DPS", "ECLIPJ2000", mock_date_time_2et.return_value)
        np.testing.assert_array_equal([12.0, 15.0, 18.0], mock_reclat.call_args_list[1][0][0])

        self.assertEqual(0.4679210985587912, call_args.spacecraft_radius)
        self.assertEqual(261.6337638953414, call_args.spacecraft_longitude)
        self.assertEqual(-51.56620156177409, call_args.spacecraft_latitude)
        self.assertEqual(vx, call_args.spacecraft_velocity_x)
        self.assertEqual(vy, call_args.spacecraft_velocity_y)
        self.assertEqual(vz, call_args.spacecraft_velocity_z)
        self.assertEqual(np.rad2deg(spin_axis_long) % 360, call_args.spin_axis_longitude)
        self.assertEqual(np.rad2deg(spin_axis_lat), call_args.spin_axis_latitude)

    def test_determine_spacecraft_info_using_predict_if_needed(self):
        with (tempfile.TemporaryDirectory() as tmp_dir):

            input_files = [
                get_integration_test_data_path("spice/de440.bsp"),
                get_integration_test_data_path("spice/imap_dps_2025_359_2026_058_002.ah.bc"),
                get_integration_test_data_path("spice/imap_dps_2025_359_2026_131_002.ah.bc"),
                get_integration_test_data_path("spice/imap_recon_20250925_20260520_v01.bsp"),
                get_integration_test_data_path("spice/imap_science_120.tf"),
                get_integration_test_data_path("spice/imap_sclk_0171.tsc"),
                get_integration_test_data_path("spice/naif0012.tls"),
            ]

            with mock_imap_data_access(Path(tmp_dir), input_files):
                spice_paths_with_predict = [SPICEFilePath(p.name) for p in input_files]
                spice_paths_without_predict = [p for p in spice_paths_with_predict if "imap_dps_2025_359_2026_131" not in p.filename.name]

                spice_dir = imap_data_access.config["DATA_DIR"] / "imap/spice"
                metakernel_with_predict = Path(tmp_dir, "metakernel_with_predict")
                metakernel_with_predict.write_text(create_metakernel(spice_dir, spice_paths_with_predict))

                metakernel_without_predict = Path(tmp_dir, "metakernel_without_predict")
                metakernel_without_predict.write_text(create_metakernel(spice_dir, spice_paths_without_predict))

                input_files_with_predict = [Path("parent/kernel 1"), Path("kernel 2"), Path("kernel 3"), Path("kernel 4")]
                input_files_without_predict = [Path("kernel 5"), Path("kernel 6"), Path("kernel 7"), Path("kernel 8")]

                spice_with_predict = FurnishMetakernelOutput(metakernel_with_predict, input_files_with_predict)
                spice_without_predict = FurnishMetakernelOutput(metakernel_without_predict, input_files_without_predict)

                date_with_predict = datetime(2026, 3, 25, 12)
                date_without_predict = datetime(2026, 1, 15, 12)

                result_without_predict = determine_spacecraft_info_using_predict_if_needed(date_without_predict, spice_with_predict, spice_without_predict)
                spacecraft_info, glows_flags, kernel_names = result_without_predict
                self.assertEqual(GlowsL3Flags.NONE, glows_flags)
                self.assertEqual(["kernel 5", "kernel 6", "kernel 7", "kernel 8"], kernel_names)
                self.assertEqual(GlowsL3eSpacecraftInfo(spacecraft_radius=0.974926812207828,
                       spacecraft_longitude=np.float64(114.98987223111648),
                       spacecraft_latitude=np.float64(-0.02732735672494243),
                       spacecraft_velocity_x=np.float64(-27.105121226323945),
                       spacecraft_velocity_y=np.float64(-12.572196725157356),
                       spacecraft_velocity_z=np.float64(0.04038850296808949),
                       spin_axis_longitude=np.float64(291.46869723545007),
                       spin_axis_latitude=np.float64(0.08089846053031882)), spacecraft_info)

                result_with_predict = determine_spacecraft_info_using_predict_if_needed(date_with_predict, spice_with_predict, spice_without_predict)
                spacecraft_info, glows_flags, kernel_names = result_with_predict
                self.assertEqual(GlowsL3Flags.PREDICTIVE_EPHEMERIS, glows_flags)
                self.assertEqual(["kernel 1", "kernel 2", "kernel 3", "kernel 4"], kernel_names)
                self.assertEqual(GlowsL3eSpacecraftInfo(spacecraft_radius=0.9869148557889134,
                       spacecraft_longitude=np.float64(184.44380384879602),
                       spacecraft_latitude=np.float64(0.04367233154056944),
                       spacecraft_velocity_x=np.float64(1.8304001531697336),
                       spacecraft_velocity_y=np.float64(-29.587138738875474),
                       spacecraft_velocity_z=np.float64(-0.009923833866874787),
                       spin_axis_longitude=np.float64(0.9116206658792297),
                       spin_axis_latitude=np.float64(0.0758987849137639)) ,spacecraft_info)

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

    def test_compute_glows_flags_for_repoint(self):
        epochs = [
            datetime(2025, 5, 6, 0, 0),
            datetime(2025, 5, 16, 0, 0),
            datetime(2025, 5, 26, 0, 0),
            datetime(2025, 6, 5, 0, 0),
            datetime(2025, 6, 15, 0, 0),
        ]
        flags = [
            GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO,
            GlowsL3Flags.NONE,
            GlowsL3Flags.NONE,
            GlowsL3Flags.PREDICTIVE_EPHEMERIS,
            GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO | GlowsL3Flags.PREDICTIVE_EPHEMERIS,
        ]

        cases = [
            ("between crs 1 and 2", datetime(2025, 5, 10), GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO),
            ("between crs 2 and 3", datetime(2025, 5, 25, 23), GlowsL3Flags.NONE),
            ("between crs 3 and 4", datetime(2025, 5, 26, 1), GlowsL3Flags.PREDICTIVE_EPHEMERIS),
            ("exactly on cr 4", datetime(2025, 6, 5), GlowsL3Flags.PREDICTIVE_EPHEMERIS),
            ("between cr 4 and 5", datetime(2025, 6, 7), GlowsL3Flags.PREDICTIVE_EPHEMERIS | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO),
            ("after last data point", datetime(2025, 6, 15, 1), GlowsL3Flags.PREDICTIVE_EPHEMERIS | GlowsL3Flags.NOMINAL_ALPHA_PROTON_RATIO)
        ]

        for name, repoint_midpoint, expected in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    cdf_path = Path(tmp_dir, "l3d.cdf")
                    with CDF(str(cdf_path), create=True) as cdf:
                        cdf["epoch"] = epochs
                        cdf.new("glows_flags", data=flags, type=const.CDF_UINT2, recVary=True)

                    actual = compute_glows_flags_for_repoint(cdf_path, repoint_midpoint)

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
        expected_repoint_numbers = set(range(3682, 3724))

        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")

        set_global_repoint_table_paths([repointing_path])
        repointing_data = get_repoint_data()

        actual_repoint_numbers = get_repoint_numbers_within_cr_window(start_cr, end_cr, repointing_data)

        np.testing.assert_array_equal(actual_repoint_numbers, expected_repoint_numbers)

    def test_get_repoint_numbers_within_cr_window_returns_empty_for_none_start(self):
        start_cr = None
        end_cr = 2094

        repointing_path = get_test_data_path("fake_1_day_repointing_file.csv")

        set_global_repoint_table_paths([repointing_path])
        repointing_data = get_repoint_data()

        actual_repoint_numbers = get_repoint_numbers_within_cr_window(start_cr, end_cr, repointing_data)

        np.testing.assert_array_equal(actual_repoint_numbers, [])


    def test_calculate_energy_deltas(self):
        centers = np.array([10, 1000, 100000])
        expected_energy_delta_minus = np.array([9, 900, 90000])
        expected_energy_delta_plus = np.array([90, 9000, 900000])

        actual_energy_delta_plus, actual_energy_delta_minus = calculate_energy_deltas(centers)

        np.testing.assert_array_equal(actual_energy_delta_plus, expected_energy_delta_plus)
        np.testing.assert_array_equal(actual_energy_delta_minus, expected_energy_delta_minus)