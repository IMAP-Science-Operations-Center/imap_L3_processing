import os
import sys
from pathlib import Path

import yaml

from imap_l3_processing import cdf


def fix_fieldnam():
    variable_yaml_paths = get_variable_yaml_paths()

    for variable_yaml in variable_yaml_paths:
        print("cleaning yaml file at ", variable_yaml)
        with open(variable_yaml, mode="r") as file:
            yaml_data = yaml.safe_load(file)
            for key, item in yaml_data.items():
                if 'NAME' not in item.keys():
                    continue
                if "_" in yaml_data[key]['FIELDNAM']:
                    yaml_data[key]['FIELDNAM'] = key.replace("_", " ").title()
                if len(key) > 30:
                    print("key over 30", variable_yaml, key)

        with open(variable_yaml, mode="w") as file:
            yaml.safe_dump(yaml_data, file, sort_keys=False, allow_unicode=True)


def get_variable_yaml_paths() -> list[Path]:
    yaml_path = Path(cdf.__file__).parent / "config"

    return [yaml_path / str(filename) for filename in os.listdir(yaml_path) if
            "variable_attrs" in str(filename)]


if __name__ == "__main__":
    if "fieldnam" in sys.argv:
        fix_fieldnam()
