import argparse
import itertools
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from tests.test_helpers import get_test_data_path


def create_survival_probabilities_file(glows_file_path: Path, date_for_file: datetime, sensor: str):
    filename = f"imap_glows_l3e_survival-probabilities-hi-{sensor}_{date_for_file.strftime('%Y%m%d')}_v001.cdf"
    cdf_file_path = get_test_data_path(f"hi/fake_l3e_survival_probabilities/{sensor}/{filename}")

    cdf_file_path.unlink(missing_ok=True)
    with open(glows_file_path) as input_data:
        energy_line = [line for line in input_data.readlines() if line.startswith("#energy_grid")]
        assert len(energy_line) == 1
        energies = np.array([float("".join(list(t))) for t in itertools.batched(energy_line[0][19:-1], 10)])
        energy_units = re.search(r'\[(.+)]', energy_line[0]).group(1)

    spin_angle_and_survival_probabilities = np.loadtxt(glows_file_path, skiprows=200)
    spin_angles = spin_angle_and_survival_probabilities[:, 0]
    survival_probabilities = spin_angle_and_survival_probabilities[:, 1:].T

    with CDF(str(cdf_file_path), '') as c:
        c.new("epoch", [date_for_file], pycdf.const.CDF_TIME_TT2000)
        c.new("epoch_delta", [12 * 60 * 60 * 1e9], pycdf.const.CDF_INT8)
        c.new("energy", energies, pycdf.const.CDF_FLOAT, recVary=False)
        c['energy'].attrs["UNITS"] = energy_units
        c.new("spin_angle", spin_angles, pycdf.const.CDF_FLOAT, recVary=False)
        c.new("probability_of_survival", np.array(survival_probabilities)[np.newaxis, ...], pycdf.const.CDF_FLOAT,
              compress=pycdf.const.GZIP_COMPRESSION)

        c["energy"].attrs["FILLVAL"] = -1e31
        c["spin_angle"].attrs["FILLVAL"] = -1e31
        c["probability_of_survival"].attrs["FILLVAL"] = -1e31

    return survival_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")

    args = parser.parse_args()

    input_dir = Path(args.input_folder)

    start_date = datetime(2025, 4, 16, hour=12)
    for i, file in enumerate(input_dir.iterdir()):
        pieces = file.name.split("_")
        date = start_date + timedelta(days=i)
        create_survival_probabilities_file(file, date, "90")
