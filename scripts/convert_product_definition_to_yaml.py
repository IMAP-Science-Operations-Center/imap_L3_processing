import argparse
import csv
import os
import subprocess
from pathlib import Path


def convert_csv_to_yaml(file_path):
    required_for_data = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "DEPEND_0", "DISPLAY_TYPE",
                         "FIELDNAM", "FORMAT", "LABLAXIS", "UNITS", "VALIDMIN", "VALIDMAX", "FILLVAL", "LABL_PTRS",
                         "UNIT_PTR", ]
    non_required_columns = ["SCALE_TYP", "SCAL_PTR", "VAR_NOTES", "TIME_BASE", "TIME_SCALE", "LEAP_SECONDS_INCLUDED",
                            "ABSOLUTE_ERROR", "AVG_TYPE", "BIN_LOCATION", "DELTA_PLUS_VAR", "DELTA_MINUS_VAR",
                            "DERIVN", "DICT_KEY", "MONOTON", "SCALEMIN", "SCALEMAX", "REFERENCE_POSITION",
                            "RELATIVE_ERROR", "RESOLUTION", "SI_CONVERSION", "DEPEND_1", "DEPEND_2", "DEPEND_3",
                            "VARIABLE_PURPOSE"]
    yaml_text = ""
    with open(file_path, "r") as f:
        csvreader = csv.reader(f)
        row_headers = next(csvreader)
        units_units_ptr_inclusion = [False, False]
        for row in csvreader:
            yaml_text += f"{row[0]}:\n"

            for index, rowitem in enumerate(row):
                if rowitem == "":
                    display_rowitem = "' '"
                else:
                    display_rowitem = rowitem

                column_is_empty_and_not_required = (rowitem == "" and row_headers[index] not in required_for_data)
                column_not_listed_in_specified_list = row_headers[index] not in (
                        required_for_data + non_required_columns)
                if column_is_empty_and_not_required or column_not_listed_in_specified_list:
                    continue

                if row_headers[index] in ["UNITS", "UNIT_PTR"]:
                    units_units_ptr_inclusion[1 if row_headers[index] == "UNITS" else 0] = True
                    if all(status == True for status in units_units_ptr_inclusion):
                        continue
                yaml_text += f"   {row_headers[index]}: {display_rowitem}\n"

    return yaml_text


def process_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--generate-cdf-for-instrument")

    args = parser.parse_args()

    yaml_text = convert_csv_to_yaml(args.csv)

    filename_without_extension = Path(args.csv).name.split(".")[0]
    print(filename_without_extension)
    yaml_file_name = f"imap_l3_processing/cdf/config/{filename_without_extension}.yaml"
    if os.path.isfile(yaml_file_name):
        if input(
                "The Yaml already exists in the config folder! Are your sure your want to overwrite the other pairs work? [Y/n]") != "n":
            os.remove(yaml_file_name)

    with open(yaml_file_name, "w") as yaml_file:
        yaml_file.write(yaml_text)

    if args.generate_cdf_for_instrument is not None:
        subprocess.run(["python", "run_local.py", args.generate_cdf_for_instrument])

# if __name__ == "__main__":
#     process_csv()
