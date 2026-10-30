import csv
import sys
from itertools import chain
from pathlib import Path

import yaml

if __name__ == "__main__":
    yaml_path = sys.argv[1]
    outpath = Path(__file__).parent.parent / "data" / f"{Path(yaml_path).name[:-4]}csv"
    with open(yaml_path) as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    all_headers = {
        key: "" for key in set(chain.from_iterable([data.keys() for data in yaml_data.values()]))
    }
    valid_variables = [data for data in yaml_data.values() if "NAME" in data]
    with open(outpath, "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=all_headers.keys())
        writer.writeheader()
        for variable in valid_variables:
            writer.writerow({**all_headers, **variable})
        print("wrote file to ", outpath)
