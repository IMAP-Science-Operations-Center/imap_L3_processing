import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

from astropy.time import Time, TimeDelta

from imap_l3_processing.glows.l3bc.dependency_validator import validate_omni2_dependency, \
    validate_dependencies, validate_f107_fluxtable_dependency, validate_lyman_alpha_dependency
from tests.test_helpers import get_test_data_path


class TestDependencyValidator(unittest.TestCase):
    def test_validate_omni2_dependency(self):
        file_path = get_test_data_path("glows/glows_omni2.dat")

        test_cases = [
            ("Missing All Values Past Buffer", datetime(1994, 7, 12), timedelta(days=7),
             False),
            ("Density 1 Exists", datetime(1994, 7, 12), timedelta(days=6), False),
            ("Speed 1 Exists", datetime(1994, 7, 12), timedelta(days=5), False),
            ("Alpha 1 Exists", datetime(1994, 7, 12), timedelta(days=4), False),
            ("Density 2 Exists", datetime(1994, 7, 12), timedelta(days=3), False),
            ("Speed 2 Exists", datetime(1994, 7, 12), timedelta(days=2), False),
            ("Missing no values", datetime(1994, 7, 12), timedelta(days=1), True),
        ]

        for name, end_date, buffer, expected in test_cases:
            with self.subTest(name):
                actual = validate_omni2_dependency(file_path=file_path, end_date_exclusive=end_date, buffer=buffer)
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
        end_date_inclusive = datetime(1994,7, 17)
        buffer = timedelta(2)
        omni_file_path = Path("omni path")
        fluxtable_file_path = Path("fluxtable path")
        lyman_alpha_file_path = Path("fluxtable path")

        for case_name, omni2_validation, flux_validation, lyman_alpha_validation, expected in test_cases:
            with self.subTest(case_name):
                mock_validate_omni2_dependency.return_value = omni2_validation
                mock_validate_f107_fluxtable_dependency.return_value = flux_validation
                mock_validate_lyman_alpha_dependency.return_value = lyman_alpha_validation

                actual = validate_dependencies(end_date_inclusive, buffer, omni_file_path,
                                               fluxtable_file_path, lyman_alpha_file_path)
                self.assertEqual(expected, actual)

                mock_validate_omni2_dependency.assert_called_once_with(end_date_inclusive, buffer,
                                                                       omni_file_path)

                mock_validate_f107_fluxtable_dependency.assert_called_once_with(end_date_inclusive, buffer,
                                                                                fluxtable_file_path)

                mock_validate_lyman_alpha_dependency.assert_called_once_with(end_date_inclusive, buffer,
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
