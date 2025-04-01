import csv
import unittest

from scripts.convert_product_definition_to_yaml import convert_csv_to_yaml


class TestConvertProductDefinitionToYamlTest(unittest.TestCase):

    def test_parse_required_for_data_with_multiple_depend(self):
        row_columns = {
            "NAME": ["intensity_by_pitch_angle_and_gyrophase", "energy_label", "pitch_angle_label"],
            "DATA_TYPE": ["float32", "str", "str"],
            "CATDESC": ["Intensity organized by pitch angle and gyrophase",
                        "energy label", "pitch angle label"],
            "VAR_TYPE": ["data", "metadata", "metadata"],
            "DEPEND_0": ["epoch", "", ""],
            "DEPEND_1": ["energy", "", ""],
            "DEPEND_2": ["pitch_angle", "", ""],
            "DISPLAY_TYPE": ["spectrogram", "", ""],
            "FIELDNAM": ["Electron Intensity", "energy label", "pitch angle label"],
            "FORMAT": ["F9.3", "a20", "a20"],
            "UNITS": ["cm^-2 sr^-1 s^-1 eV^-1", "", ""],
            "VALIDMIN": ["1E-40", "", ""],
            "VALIDMAX": ["", "", ""],
            "FILLVAL": ["-1.00E+31", "", ""],
            "LABL_PTR_1": ["energy_label", "", ""],
            "LABL_PTR_2": ["pitch_angle_label", "", ""],
            "VARIABLE_PURPOSE": ["primary var, summary", "", ""]
        }

        rows = [
            [k for k in row_columns.keys()],
            [v[0] for v in row_columns.values()],
            [v[1] for v in row_columns.values()],
            [v[2] for v in row_columns.values()],
        ]

        filePath = "csv_test_file.csv"
        with open(filePath, "w", newline='') as csvfile:
            test_csv_writer = csv.writer(csvfile)
            test_csv_writer.writerows(rows)

        expected_yaml = """intensity_by_pitch_angle_and_gyrophase:
   NAME: intensity_by_pitch_angle_and_gyrophase
   DATA_TYPE: float32
   CATDESC: Intensity organized by pitch angle and gyrophase
   VAR_TYPE: data
   DEPEND_0: epoch
   DEPEND_1: energy
   DEPEND_2: pitch_angle
   DISPLAY_TYPE: spectrogram
   FIELDNAM: Electron Intensity
   FORMAT: F9.3
   UNITS: cm^-2 sr^-1 s^-1 eV^-1
   VALIDMIN: 1E-40
   VALIDMAX: ' '
   FILLVAL: -1.00E+31
   LABL_PTR_1: energy_label 
   LABL_PTR_2: pitch_angle_label
   VARIABLE_PURPOSE: primary var, summary
energy_label:
   NAME: energy_label
   DATA_TYPE: str
   CATDESC: energy label
   VAR_TYPE: metadata
   FIELDNAM: energy label
   FORMAT: a20
pitch_angle_label:
   NAME: pitch_angle_label
   DATA_TYPE: str
   CATDESC: pitch angle label
   VAR_TYPE: metadata
   FIELDNAM: pitch angle label
   FORMAT: a20"""
        actual_yaml = convert_csv_to_yaml(filePath)

        self.assertEqual(expected_yaml, actual_yaml)

    def test_parse_required_for_data_only_depend_0(self):
        required_for_data = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "RECORD_VARYING", "DEPEND_0", "DISPLAY_TYPE",
                             "FIELDNAM", "FORMAT", "LABLAXIS", "UNITS", "VALIDMIN", "VALIDMAX", "FILLVAL"]

        test_csv_values = [
            "core_fit_num_points",
            "float32",
            "Number of energies used on core fit",
            "support_data",
            "RV",
            "epoch",
            "no_plot",
            "",
            "",
            "",
            "",
            "",
            "",
            "-1.00E+31"
        ]
        filePath = "csv_test_file.csv"
        with open(filePath, "w", newline='') as csvfile:
            test_csv_writer = csv.writer(csvfile)
            test_csv_writer.writerow(required_for_data)
            test_csv_writer.writerow(test_csv_values)

            expected_yaml = """core_fit_num_points:
   NAME: core_fit_num_points
   DATA_TYPE: float32
   CATDESC: Number of energies used on core fit
   VAR_TYPE: support_data
   RECORD_VARYING: RV
   DEPEND_0: epoch
   FIELDNAM: ' '
   FORMAT: ' '
   LABLAXIS: ' '
   UNITS: ' '
   VALIDMIN: ' '
   VALIDMAX: ' '
   FILLVAL: -1.00E+31"""
        actual_yaml = convert_csv_to_yaml(filePath)
        self.assertEqual(expected_yaml, actual_yaml)

    def test_parse_required_for_support_data_non_rv(self):
        required_for_data = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "RECORD_VARYING", "DISPLAY_TYPE",
                             "FIELDNAM", "FORMAT", "LABLAXIS", "UNITS", "VALIDMIN", "VALIDMAX", "FILLVAL"]

        test_csv_values = [
            "energy",
            "float32",
            "energy bins",
            "support_data",
            "NRV",
            "no_plot",
            "",
            "",
            "",
            "",
            "",
            "",
            "-1.00E+31"
        ]
        filePath = "csv_test_file.csv"
        with open(filePath, "w", newline='') as csvfile:
            test_csv_writer = csv.writer(csvfile)
            test_csv_writer.writerow(required_for_data)
            test_csv_writer.writerow(test_csv_values)

            expected_yaml = """energy:
   NAME: energy
   DATA_TYPE: float32
   CATDESC: energy bins
   VAR_TYPE: support_data
   RECORD_VARYING: NRV
   FIELDNAM: ' '
   FORMAT: ' '
   LABLAXIS: ' '
   UNITS: ' '
   VALIDMIN: ' '
   VALIDMAX: ' '
   FILLVAL: -1.00E+31"""
        actual_yaml = convert_csv_to_yaml(filePath)
        self.assertEqual(expected_yaml, actual_yaml)

    def test_parse_required_for_data_epoch(self):
        required_for_data = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "DEPEND_0", "DISPLAY_TYPE",
                             "FIELDNAM", "FORMAT", "LABLAXIS", "UNITS", "VALIDMIN", "VALIDMAX", "FILLVAL"]

        test_csv_values = [
            "epoch",
            "float32",
            "The time",
            "support_data",
            "",
            "no_plot",
            "",
            "",
            "",
            "",
            "",
            "",
            "-1.00E+31"
        ]
        filePath = "csv_test_file.csv"
        with open(filePath, "w", newline='') as csvfile:
            test_csv_writer = csv.writer(csvfile)
            test_csv_writer.writerow(required_for_data)
            test_csv_writer.writerow(test_csv_values)

            expected_yaml = """epoch:
   NAME: epoch
   DATA_TYPE: float32
   CATDESC: The time
   VAR_TYPE: support_data
   FIELDNAM: ' '
   FORMAT: ' '
   LABLAXIS: ' '
   UNITS: ' '
   VALIDMIN: ' '
   VALIDMAX: ' '
   FILLVAL: -1.00E+31"""
        actual_yaml = convert_csv_to_yaml(filePath)
        self.assertEqual(expected_yaml, actual_yaml)

    def test_parse_does_not_include_columns_if_they_are_blank(self):
        all_headers = ["NAME",
                       "DATA_TYPE",
                       "Data shape",
                       "CATDESC",
                       "VAR_TYPE",
                       "RECORD_VARYING",
                       "DEPEND_0",
                       "DEPEND_1",
                       "DEPEND_2",
                       "DEPEND_3",
                       "DISPLAY_TYPE",
                       "FIELDNAM",
                       "FORMAT",
                       "LABLAXIS",
                       "UNITS",
                       "VALIDMIN",
                       "VALIDMAX",
                       "FILLVAL",
                       "LABL_PTRS",
                       "UNIT_PTR",
                       "SCALE_TYP",
                       "SCAL_PTR",
                       "VAR_NOTES",
                       "TIME_BASE",
                       "TIME_SCALE",
                       "LEAP_SECONDS_INCLUDED",
                       "ABSOLUTE_ERROR",
                       "AVG_TYPE",
                       "BIN_LOCATION",
                       "DELTA_PLUS_VAR",
                       "DELTA_MINUS_VAR",
                       "DERIVN",
                       "DICT_KEY",
                       "MONOTON",
                       "SCALEMIN",
                       "SCALEMAX",
                       "REFERENCE_POSITION",
                       "RELATIVE_ERROR",
                       "RESOLUTION",
                       "SI_CONVERSION"
                       ]

        row_contents = [
            "stim_tag",
            "int8",
            "(epoch, stim_tag)",
            "Stim Tag (stimulus event)",
            "support_data",
            "RV",
            "epoch",
            "",
            "",
            "",
            "no_plot",
            "stim_tag",
            "",
            "",
            "",
            "0",
            "1",
            "-1.28E+02",
            "",
            "",
            "linear",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]

        filePath = "csv_test_file.csv"
        with open(filePath, "w", newline='') as csvfile:
            test_csv_writer = csv.writer(csvfile)
            test_csv_writer.writerow(all_headers)
            test_csv_writer.writerow(row_contents)

        expected_yaml = """stim_tag:
   NAME: stim_tag
   DATA_TYPE: int8
   CATDESC: Stim Tag (stimulus event)
   VAR_TYPE: support_data
   RECORD_VARYING: RV
   DEPEND_0: epoch
   FIELDNAM: stim_tag
   FORMAT: ' '
   LABLAXIS: ' '
   UNITS: ' '
   VALIDMIN: 0
   VALIDMAX: 1
   FILLVAL: -1.28E+02
   SCALE_TYP: linear"""

        actual_yaml = convert_csv_to_yaml(filePath)
        self.assertEqual(expected_yaml, actual_yaml)
