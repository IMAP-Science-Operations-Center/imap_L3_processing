import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from astropy.time import Time

from imap_l3_processing.glows.l3b.dependency_validator import validate_omni2_dependency, \
    validate_dependencies, validate_f107_fluxtable_dependency, validate_lyman_alpha_dependency
from tests.test_helpers import get_test_data_path


class TestDependencyValidator(unittest.TestCase):
    def test_validate_omni2_dependency(self):
        file_path = get_test_data_path("glows/glows_omni2.dat")

        test_cases = [
            ("happy case", Time("1994:188:00:00:00", format="yday"), Time("1994:191:00:00:00", format="yday"), True),
            ("desnsity column 1 fill value", Time("1994:193:00:00:00", format="yday"),
             Time("1994:196:00:00:00", format="yday"), False),
            ("desnsity column 2 fill value", Time("1994:195:00:00:00", format="yday"),
             Time("1994:198:00:00:00", format="yday"), False),
            ("speed column 1 fill value", Time("1994:197:00:00:00", format="yday"),
             Time("1994:200:00:00:00", format="yday"), False),
            ("speed column 2 fill value", Time("1994:199:00:00:00", format="yday"),
             Time("1994:202:00:00:00", format="yday"), False),
            ("alpha column 1 fill value", Time("1994:201:00:00:00", format="yday"),
             Time("1994:204:00:00:00", format="yday"), False),
            ("alpha column 2 fill value", Time("1994:203:00:00:00", format="yday"),
             Time("1994:206:00:00:00", format="yday"), False),
            ("missing end values", Time("1994:206:00:00:00", format="yday"), Time("1994:208:00:00:00", format="yday"),
             False),
        ]

        for name, start_date, end_date, expected in test_cases:
            with self.subTest(name):
                actual = validate_omni2_dependency(file_path=file_path, start_date_inclusive=start_date,
                                                   end_date_exclusive=end_date)
                self.assertEqual(expected, actual)

    @patch("imap_l3_processing.glows.l3b.dependency_validator.validate_lyman_alpha_dependency")
    @patch("imap_l3_processing.glows.l3b.dependency_validator.validate_f107_fluxtable_dependency")
    @patch("imap_l3_processing.glows.l3b.dependency_validator.validate_omni2_dependency")
    def test_validate_dependencies(self, mock_validate_omni2_dependency: Mock,
                                   mock_validate_f107_fluxtable_dependency: Mock,
                                   mock_validate_lyman_alpha_dependency: Mock):
        test_cases = [
            ("happy case", True, True, True, True),
            ("omni validation fails", False, True, True, False),
            ("flux validation fails", True, False, True, False),
            ("lyman validation fails", True, True, False, False)
        ]
        start_date_inclusive = Time("1994:188:00:00:00", format="yday")
        end_date_inclusive = Time("1994:198:00:00:00", format="yday")
        omni_file_path = Path("omni path")
        fluxtable_file_path = Path("fluxtable path")
        lyman_alpha_file_path = Path("fluxtable path")

        for case_name, omni2_validation, flux_validation, lyman_alpha_validation, expected in test_cases:
            with self.subTest(case_name):
                mock_validate_omni2_dependency.return_value = omni2_validation
                mock_validate_f107_fluxtable_dependency.return_value = flux_validation
                mock_validate_lyman_alpha_dependency.return_value = lyman_alpha_validation

                actual = validate_dependencies(start_date_inclusive, end_date_inclusive, omni_file_path,
                                               fluxtable_file_path, lyman_alpha_file_path)
                self.assertEqual(expected, actual)

                mock_validate_omni2_dependency.assert_called_once_with(start_date_inclusive, end_date_inclusive,
                                                                       omni_file_path)

                mock_validate_f107_fluxtable_dependency.assert_called_once_with(start_date_inclusive,
                                                                                end_date_inclusive,
                                                                                fluxtable_file_path)

                mock_validate_lyman_alpha_dependency.assert_called_once_with(end_date_inclusive,
                                                                             lyman_alpha_file_path)
                mock_validate_omni2_dependency.reset_mock()
                mock_validate_f107_fluxtable_dependency.reset_mock()
                mock_validate_lyman_alpha_dependency.reset_mock()

    def test_validate_f107_dependency(self):
        file_path = get_test_data_path("glows/glows_f107_fluxtable.txt")
        test_cases = [
            ("happy case", Time("2025-02-23 00:00:00"), Time("2025-02-25 00:00:00"), True),
            ("not enough data end", Time("2025-02-25 00:00:00"), Time("2025-02-27 00:00:00"), False),
            ("not enough data start", Time("2025-02-21 00:00:00"), Time("2025-02-25 00:00:00"),
             False)]
        for name, start_date, end_date, expected in test_cases:
            with self.subTest(name):
                is_validated = validate_f107_fluxtable_dependency(file_path=file_path, start_date_inclusive=start_date,
                                                                  end_date_exclusive=end_date)

                self.assertEqual(expected, is_validated)

    def test_validate_lyman_alpha_dependency(self):
        file_path = get_test_data_path("glows/lyman_alpha_composite.nc")
        test_cases = [
            ("happy case", Time("2025-03-20 00:00:00"), True),
            ("not enough data end", Time("2025-03-22 00:00:00"), False)
        ]
        for name, end_date, exptected in test_cases:
            with self.subTest(name):
                is_validated = validate_lyman_alpha_dependency(end_date_exclusive=end_date, file_path=file_path)
                self.assertEqual(exptected, is_validated)
