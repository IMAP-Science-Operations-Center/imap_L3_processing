import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock

from astropy.time import Time, TimeDelta

from imap_l3_processing.glows.l3bc.dependency_validator import validate_omni2_dependency, \
    validate_dependencies, validate_f107_fluxtable_dependency, validate_lyman_alpha_dependency
from tests.test_helpers import get_test_data_path


class TestDependencyValidator(unittest.TestCase):
    def test_validate_omni2_dependency_returns_true_when_alpha_fill_but_density_and_speed_valid(self):
        with TemporaryDirectory() as tmp:
            omni_path = Path(tmp) / "omni2_slice.txt"
            row = ["0"] * 35
            row[0] = "2000"
            row[1] = "32"
            row[2] = "0"
            row[5] = "1"
            row[23] = "5.0"
            row[24] = "400.0"
            row[27] = "99.999"
            row[30] = "0.1"
            row[31] = "1.0"
            row[34] = "99.999"
            line = " ".join(row) + "\n"
            omni_path.write_text(line + line)
            self.assertTrue(
                validate_omni2_dependency(
                    cr_start_date=datetime(2000, 2, 1),
                    cr_end_date_exclusive=datetime(2000, 2, 2),
                    file_path=omni_path,
                )
            )

    def test_validate_omni2_dependency(self):
        file_path = get_test_data_path("glows/glows_omni2.dat")

        test_cases = [
            ("Missing no values", datetime(1994, 7, 13), datetime(1994, 7, 14), True),
            ("Speed 2 Exists", datetime(1994, 7, 14), datetime(1994, 7, 15), True),
            ("Density 2 Exists", datetime(1994, 7, 15), datetime(1994, 7, 16), False),
            ("Alpha 1 Exists", datetime(1994, 7, 16), datetime(1994, 7, 17), False),
            ("Speed 1 Exists", datetime(1994, 7, 17), datetime(1994, 7, 18), False),
            ("Density 1 Exists", datetime(1994, 7, 18), datetime(1994, 7, 19), False),
            ("Missing All Values", datetime(1994, 7, 19), datetime(1994, 7, 20), False),
            ("Empty window", datetime(1994, 7, 20), datetime(1994, 7, 21), False),
        ]

        for name, cr_start_date, cr_end_date_exclusive, expected in test_cases:
            with self.subTest(name):
                actual = validate_omni2_dependency(file_path=file_path,
                                                   cr_start_date=cr_start_date,
                                                   cr_end_date_exclusive=cr_end_date_exclusive)
                self.assertEqual(expected, actual)

    @patch("imap_l3_processing.glows.l3bc.dependency_validator.validate_lyman_alpha_dependency")
    @patch("imap_l3_processing.glows.l3bc.dependency_validator.validate_f107_fluxtable_dependency")
    @patch("imap_l3_processing.glows.l3bc.dependency_validator.validate_omni2_dependency")
    def test_validate_dependencies(self, mock_validate_omni2_dependency: Mock,
                                   mock_validate_f107_fluxtable_dependency: Mock,
                                   mock_validate_lyman_alpha_dependency: Mock):
        test_cases = [
            ("happy case", True, True, True, True),
            ("omni validation fails", False, True, True, False),
            ("flux validation fails", True, False, True, False),
            ("lyman validation fails", True, True, False, False)
        ]
        cr_start_date = datetime(1994, 7, 10)
        cr_end_date = datetime(1994, 7, 17)
        buffer = timedelta(2)
        omni_file_path = Path("omni path")
        fluxtable_file_path = Path("fluxtable path")
        lyman_alpha_file_path = Path("fluxtable path")

        for case_name, omni2_validation, flux_validation, lyman_alpha_validation, expected in test_cases:
            with self.subTest(case_name):
                mock_validate_omni2_dependency.return_value = omni2_validation
                mock_validate_f107_fluxtable_dependency.return_value = flux_validation
                mock_validate_lyman_alpha_dependency.return_value = lyman_alpha_validation

                actual = validate_dependencies(cr_start_date, cr_end_date, buffer, omni_file_path,
                                               fluxtable_file_path, lyman_alpha_file_path)
                self.assertEqual(expected, actual)

                mock_validate_omni2_dependency.assert_called_once_with(cr_start_date, cr_end_date,
                                                                       omni_file_path)

                mock_validate_f107_fluxtable_dependency.assert_called_once_with(cr_end_date, buffer,
                                                                                fluxtable_file_path)

                mock_validate_lyman_alpha_dependency.assert_called_once_with(cr_end_date, buffer,
                                                                             lyman_alpha_file_path)
                mock_validate_omni2_dependency.reset_mock()
                mock_validate_f107_fluxtable_dependency.reset_mock()
                mock_validate_lyman_alpha_dependency.reset_mock()

    def test_validate_f107_dependency(self):
        file_path = get_test_data_path("glows/glows_f107_fluxtable.txt")
        test_cases = [
            ("happy case", datetime(2025, 2, 23), timedelta(days=3), True),
            ("not enough data end", datetime(2025, 2, 23), timedelta(days=5), False),
        ]
        for name, end_date, buffer, expected in test_cases:
            with self.subTest(name):
                is_validated = validate_f107_fluxtable_dependency(file_path=file_path,
                                                                  end_date=end_date,
                                                                  buffer=buffer)

                self.assertEqual(expected, is_validated)

    def test_validate_lyman_alpha_dependency(self):
        file_path = get_test_data_path("glows/lyman_alpha_composite.nc")
        test_cases = [
            ("happy case", datetime(2025,3,16), timedelta(4), True),
            ("not enough data end", datetime(2025,3,18), timedelta(4), False)
        ]
        for name, end_date, buffer, expected in test_cases:
            with self.subTest(name):
                is_validated = validate_lyman_alpha_dependency(end_date=end_date, buffer=buffer, file_path=file_path)
                self.assertEqual(expected, is_validated)
