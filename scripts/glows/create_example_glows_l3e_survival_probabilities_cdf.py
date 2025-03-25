import re
from datetime import datetime
from os import unlink
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

path = Path(__file__)
input_path = path.parent.parent.parent / "instrument_team_data" / "glows" / "probSur.Imap.Hi_2009.000_135.0.txt"
cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "glows" / "imap_glows_l3e_survival-probabilities-hi_20250324_v001.cdf"

cdf_file_path.unlink(missing_ok=True)

with open(input_path) as input_data:
    energy_line = [line for line in input_data.readlines() if line.startswith("#energy_grid")]
    assert len(energy_line) == 1
    match = [float(x) for x in re.findall(r"\s+([0-9.]+)", energy_line[0])]
    energy_units = re.search(r'\[(.+)]', energy_line[0]).group(1)

    energies = np.array(match)

spin_angle_and_survival_probabilities = np.loadtxt(input_path, skiprows=200)
spin_angles = spin_angle_and_survival_probabilities[:, 0]
survival_probabilities = spin_angle_and_survival_probabilities[:, 1:]

with CDF(str(cdf_file_path), '') as c:
    c.new("epoch", [datetime(2025, 3, 24, 12)], pycdf.const.CDF_TIME_TT2000)
    c.new("epoch_delta", [12 * 60 * 60 * 1e9], pycdf.const.CDF_INT8)
    c.new("energy", energies, pycdf.const.CDF_FLOAT, recVary=False)
    c['energy'].attrs["UNITS"] = energy_units
    c.new("spin_angle", spin_angles, pycdf.const.CDF_FLOAT, recVary=False)
    c.new("probability_of_survival", survival_probabilities, pycdf.const.CDF_FLOAT)
