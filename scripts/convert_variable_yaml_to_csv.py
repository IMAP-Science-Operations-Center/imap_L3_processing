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

    all_headers = {key: '' for key in
                   set(chain.from_iterable([yaml_data[variable].keys() for variable in yaml_data.keys()]))}
    valid_variables = [yaml_data[var] for var in yaml_data.keys() if 'NAME' in yaml_data[var]]
    with open(outpath, "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=all_headers.keys())
        writer.writeheader()
        for variable in valid_variables:
            writer.writerow({**all_headers, **variable})
