import argparse
import csv
import os
import subprocess
from pathlib import Path


def convert_csv_to_yaml(file_path):
    required_for_support_data = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "DEPEND_0", "FIELDNAM", "FORMAT", "UNITS",
                                 "VALIDMIN", "VALIDMAX", "FILLVAL", "RECORD_VARYING"]

    required_for_data = required_for_support_data + ["DISPLAY_TYPE"]

    required_for_metadata = ["NAME", "DATA_TYPE", "CATDESC", "VAR_TYPE", "VAR_TYPE", "FIELDNAM", "FORMAT", ]

    non_required_columns = ["SCALE_TYP", "SCAL_PTR", "VAR_NOTES", "TIME_BASE", "TIME_SCALE", "LEAP_SECONDS_INCLUDED",
                            "ABSOLUTE_ERROR", "AVG_TYPE", "BIN_LOCATION", "DELTA_PLUS_VAR", "DELTA_MINUS_VAR",
                            "DERIVN", "DICT_KEY", "MONOTON", "SCALEMIN", "SCALEMAX", "REFERENCE_POSITION",
                            "RELATIVE_ERROR", "RESOLUTION", "SI_CONVERSION", "DEPEND_1", "DEPEND_2", "DEPEND_3",
                            "VARIABLE_PURPOSE", "UNIT_PTR", "LABL_PTR_1", "LABL_PTR_2", "LABL_PTR_3", "LABLAXIS"]
    yaml_text = ""
    try:
        with open(file_path, "r") as f:
            csvreader = csv.reader(f)
            row_headers = next(csvreader)
            header_to_index = {header: index for index, header in enumerate(row_headers)}
            for row in csvreader:
                yaml_text += f"{row[0]}:\n"

                if row[header_to_index["NAME"]].lower() == "epoch":
                    required_variables = [variable for variable in required_for_support_data if
                                          variable != "DEPEND_0"]
                elif row[header_to_index["VAR_TYPE"]] == "support_data":
                    required_variables = required_for_support_data
                elif row[header_to_index["VAR_TYPE"]] == "metadata":
                    required_variables = required_for_metadata
                else:
                    required_variables = required_for_data

                if row[header_to_index["RECORD_VARYING"]].lower() == "nrv":
                    required_variables = [variable for variable in required_variables if
                                          variable != "DEPEND_0"]

                for index, rowitem in enumerate(row):
                    if rowitem == "":
                        display_rowitem = "' '"
                    else:
                        display_rowitem = rowitem

                    column_is_empty_and_not_required = (rowitem == "" and row_headers[index] not in required_variables)
                    column_not_listed_in_specified_list = row_headers[index] not in (
                            required_variables + non_required_columns)
                    if column_is_empty_and_not_required or column_not_listed_in_specified_list:
                        continue

                    yaml_text += f"   {row_headers[index]}: {display_rowitem}\n"
    except Exception as e:
        print(e)

    yaml_text = yaml_text.strip()

    return yaml_text


def process_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument")
    parser.add_argument("--generate-cdf-for-instrument")

    args = parser.parse_args()

    instrument_to_file_path = {
        "swe": "imap_l3_processing/cdf/data_product_definition_csv/imap_swe_l3_variable_attrs.csv",
        "glows-l3a": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3a_variable_attrs.csv",
        "glows-l3b": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3b_variable_attrs.csv",
        "glows-l3c": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3c_variable_attrs.csv",
        "swapi-l3a": "imap_l3_processing/cdf/data_product_definition_csv/imap_swapi_l3a_variable_attrs.csv",
        "swapi-l3b": "imap_l3_processing/cdf/data_product_definition_csv/imap_swapi_l3b_variable_attrs.csv",
        "hit-l3-direct-event": "imap_l3_processing/cdf/data_product_definition_csv/imap_hit_l3_direct-events_variable_attrs.csv",
        "hit-l3-macropixel": "imap_l3_processing/cdf/data_product_definition_csv/imap_hit_l3_macropixel_variable_attrs.csv",
        "hi-l3-survival-probability": "imap_l3_processing/cdf/data_product_definition_csv/imap_hi_l3_h90-sf-sp-hae-4deg-6mo_variable_attrs.csv",
        "hi-l3-spectral-index": "imap_l3_processing/cdf/data_product_definition_csv/imap_hi_l3_h90-hf-sp-hae-4deg-6mo-spectral_variable_attrs.csv",
        "glows-l3e-lo": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3e_survival-probability-lo_variable_attrs.csv",
        "glows-l3e-hi-45": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3e_survival-probability-hi-45_variable_attrs.csv",
        "glows-l3e-hi-90": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3e_survival-probability-hi-90_variable_attrs.csv",
        "glows-l3e-ul": "imap_l3_processing/cdf/data_product_definition_csv/imap_glows_l3e_survival-probability-ul_variable_attrs.csv",
    }
    file_path = instrument_to_file_path[args.instrument]

    yaml_text = convert_csv_to_yaml(file_path)

    filename_without_extension = Path(file_path).name.split(".")[0]
    print(filename_without_extension)
    yaml_file_name = f"imap_l3_processing/cdf/config/{filename_without_extension}.yaml"
    if os.path.isfile(yaml_file_name):
        if input(
                "The Yaml already exists in the config folder! Are you sure you want to overwrite the other pairs work? [Y/n]") != "n":
            os.remove(yaml_file_name)

    with open(yaml_file_name, "w") as yaml_file:
        yaml_file.write(yaml_text)

    if args.generate_cdf_for_instrument is not None:
        subprocess.run(["python", "run_local.py", args.generate_cdf_for_instrument])


if __name__ == "__main__":
    process_csv()
