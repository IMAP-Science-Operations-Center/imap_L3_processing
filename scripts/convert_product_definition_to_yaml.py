import argparse
import csv
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("--generate-cdf-for-instrument")

args = parser.parse_args()

yaml_text = ""
with open(args.csv) as csvfile:
    csv_reader = csv.reader(csvfile)

    headers = next(csv_reader)
    metadata = list(csv_reader)
    for row in metadata:
        variable_name = row[0]
        metadata_strs = ["  " + f"{cdf_metadata_name}: {cdf_metadata_value}" for cdf_metadata_name, cdf_metadata_value
                         in
                         list(zip(headers, row))[3:] if cdf_metadata_value]
        yaml_text += "\n".join([f"{variable_name}:"] + metadata_strs + [""]) + "\n"

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
