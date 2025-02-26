import csv
import sys

if len(sys.argv) == 2:
    product_definition_file = sys.argv[1]
else:
    product_definition_file = "data_product_spec_csvs/imap_hit_l3_pitch_angle.csv"

with open(product_definition_file) as csvfile:
    csv_reader = csv.reader(csvfile)

    headers = next(csv_reader)
    metadata = list(csv_reader)
    for row in metadata:
        variable_name = row[0]
        metadata_strs = ["  " + f"{cdf_metadata_name}: {cdf_metadata_value}" for cdf_metadata_name, cdf_metadata_value
                         in
                         list(zip(headers, row))[3:] if cdf_metadata_value]
        variable_section = "\n".join([f"{variable_name}:"] + metadata_strs + [""])
        print(variable_section)
