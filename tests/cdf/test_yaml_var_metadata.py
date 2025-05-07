import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest import TestCase

import yaml

from imap_l3_processing import cdf


class TestCdfUtils(TestCase):

    def setUp(self):
        yaml_path = Path(cdf.__file__).parent / "config"

        variable_attrs_filenames = [str(filename) for filename in os.listdir(yaml_path) if
                                    "variable_attrs" in str(filename)]
        self.test_cases_variable = []
        for filename in variable_attrs_filenames:
            with open(yaml_path / filename) as file:
                yaml_data = yaml.safe_load(file)
                for variable_key, variable in yaml_data.items():
                    if 'NAME' in variable:
                        self.test_cases_variable.append((filename, yaml_data, variable_key, variable))

        self.test_cases_file = []
        for filename in variable_attrs_filenames:
            with open(yaml_path / filename) as file:
                yaml_data = yaml.safe_load(file)
                self.test_cases_file.append((filename, yaml_data))

    def test_variable_name_equals_key(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(msg=f"{filename}:{variable_key}"):
                self.assertEqual(variable_key, variable['NAME'],
                                 f"{filename}:{variable_key} variable key and NAME do not match")

    def test_epoch(self):
        for filename, yaml_data in self.test_cases_file:
            with self.subTest(filename):
                self.assertIn('epoch', yaml_data.keys(), "no entry in file for 'epoch'")
                self.assertEqual('epoch', yaml_data['epoch']['NAME'], "epoch must be lowercase")

    def test_delta_vars_have_same_units(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(msg=f"{filename}:{variable_key}"):
                if 'DELTA_PLUS_VAR' in variable:
                    self.assertEqual(variable['UNITS'], yaml_data[variable['DELTA_PLUS_VAR']]['UNITS'])
                if 'DELTA_MINUS_VAR' in variable:
                    self.assertEqual(variable['UNITS'], yaml_data[variable['DELTA_MINUS_VAR']]['UNITS'])

    def test_lablaxis_is_defined_for_required_types(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if 'NAME' in variable and not variable['VAR_TYPE'] == 'metadata' and not variable.get(
                        "DISPLAY_TYPE") == 'no_plot':
                    self.assertIn('LABLAXIS', variable.keys(), f'LABLAXIS should exist for {variable_key}')

    def test_var_type_exists(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                self.assertIn('VAR_TYPE', variable.keys(), f'VAR_TYPE should exist for {variable_key}')

    @unittest.skip('skipping because scaletyp/scaleptr are not required because linear is assumed')
    def test_each_data_variable_has_scaletyp_or_scaleptr(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if variable['VAR_TYPE'] == 'support_data':
                    self.assertIn('SCALETYP', variable.keys(),
                                  f'SCALETYP should exist for support_data variable {variable_key}')
                if variable['VAR_TYPE'] == 'data':
                    if 'SCALEPTR' not in variable.keys() and 'SCALETYP' not in variable.keys():
                        self.assertFalse(True, f'SCALETYP or SCALEPTR should exist for data variable {variable_key}')
                    if 'SCALEPTR' in variable.keys() and 'SCALETYP' in variable.keys():
                        self.assertFalse(True,
                                         f"SCALETYP and SCALEPTR  shouldn't both exist for data variable {variable_key}")

    def test_each_data_variable_has_a_display_type(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if variable['VAR_TYPE'] == 'data':
                    self.assertIn('DISPLAY_TYPE', variable.keys(),
                                  f'DISPLAY_TYPE should exist for data variables {variable_key}')

    def test_variable_purpose_exists_for_data_variables_and_empty_for_others(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                if variable['VAR_TYPE'] == 'data':
                    self.assertIn('VARIABLE_PURPOSE', variable.keys())

    @unittest.skip('only labels do not have fill values currently')
    def test_each_variable_has_data_type_and_fill_value(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                self.assertIn("DATA_TYPE", variable.keys(), f'{variable_key} requires "DATA_TYPE"')
                self.assertIn("FILLVAL", variable.keys(), f'{variable_key} requires "FILLVAL"')

    @unittest.skip('only labels do not have fill values currently')
    def test_variable_fill_value_matches_data_type(self):
        for filename, yaml_data, variable_key, variable in self.test_cases_variable:
            with self.subTest(f"{filename}:{variable_key}"):
                fill_val = variable.get('FILLVAL')
                match variable['DATA_TYPE']:
                    case "CDF_FLOAT":
                        self.assertEqual(fill_val, -1e31)
                    case "CDF_REAL4":
                        self.assertEqual(fill_val, -1e31)
                    case "CDF_UINT1":
                        self.assertEqual(fill_val, 255)
                    case "CDF_UINT2":
                        self.assertEqual(fill_val, 65535)
                    case "CDF_UINT4":
                        self.assertEqual(fill_val, 4294967295)
                    case "CDF_INT1":
                        self.assertEqual(fill_val, -128)
                    case "CDF_INT2":
                        self.assertEqual(fill_val, -32768)
                    case "CDF_INT4":
                        self.assertEqual(fill_val, -2147483648)
                    case "CDF_INT8":
                        self.assertEqual(fill_val, -9223372036854775808)
                    case "CDF_CHAR":
                        self.assertEqual(fill_val, ' ')
                    case "CDF_TIME_TT2000":
                        self.assertEqual(fill_val, datetime.fromisoformat('9999-12-31T23:59:59.999999999'))
                    case _:
                        self.assertFalse(True, f"Found unknown DATA_TYPE: {variable['DATA_TYPE']}")
