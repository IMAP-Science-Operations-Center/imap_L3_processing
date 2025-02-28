import csv
import sys

if len(sys.argv) == 2:
    product_definition_file = sys.argv[1]
else:
    product_definition_file = "../data_product_spec_csvs/imap_hit_l3_direct_event_product_definitions.csv"

yaml_text = ""

with open(product_definition_file) as csvfile:
    csv_reader = csv.reader(csvfile)

    headers = next(csv_reader)
    metadata = list(csv_reader)
    for row in metadata:
        variable_name = row[0]
        metadata_strs = ["  " + f"{cdf_metadata_name}: {cdf_metadata_value}" for cdf_metadata_name, cdf_metadata_value
                         in
                         list(zip(headers, row))[3:] if cdf_metadata_value]
        yaml_text += "\n".join([f"{variable_name}:"] + metadata_strs + [""]) + "\n"

with open("variable.yaml", "w") as yaml_file:
    yaml_file.write(yaml_text)
