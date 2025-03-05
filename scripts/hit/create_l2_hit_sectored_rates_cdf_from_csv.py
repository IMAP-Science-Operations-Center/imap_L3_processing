import json
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.constants import FIVE_MINUTES_IN_NANOSECONDS


def create_l2_hit_sectored_rates_cdf_from_csv(csv_file_path: str, output_filename: str):
    with CDF(output_filename, '') as cdf:
        cdf.col_major(True)
        raw_data = np.loadtxt(csv_file_path, delimiter=",", skiprows=1,
                              usecols=range(1, 121))
        flux_data = raw_data.reshape((-1, 8, 15))
        epoch_count = 1
        start_time = datetime(2025, 1, 1, 0, 5)
        epoch_data = np.array([start_time])
        epoch_delta = np.full(epoch_count, FIVE_MINUTES_IN_NANOSECONDS)

        hydrogen_data = flux_data[np.newaxis, 0:3]
        helium_data = flux_data[np.newaxis, 3:5]
        cno_data = flux_data[np.newaxis, 5:7]
        nemgsi_data = flux_data[np.newaxis, 7:9]
        iron_data = flux_data[np.newaxis, 9:10]
        cdf["hydrogen"] = hydrogen_data
        cdf["DELTA_MINUS_HYDROGEN"] = hydrogen_data - hydrogen_data * 0.09
        cdf["DELTA_PLUS_HYDROGEN"] = hydrogen_data + hydrogen_data * 0.11
        cdf["helium4"] = helium_data
        cdf["DELTA_MINUS_HELIUM4"] = helium_data - helium_data * 0.09
        cdf["DELTA_PLUS_HELIUM4"] = helium_data + helium_data * 0.11
        cdf["CNO"] = cno_data
        cdf["DELTA_MINUS_CNO"] = cno_data - cno_data * 0.09
        cdf["DELTA_PLUS_CNO"] = cno_data + cno_data * 0.11
        cdf["NeMgSi"] = nemgsi_data
        cdf["DELTA_MINUS_NEMGSI"] = nemgsi_data - nemgsi_data * 0.09
        cdf["DELTA_PLUS_NEMGSI"] = nemgsi_data + nemgsi_data * 0.11
        cdf["iron"] = iron_data
        cdf["DELTA_MINUS_IRON"] = iron_data - iron_data * 0.09
        cdf["DELTA_PLUS_IRON"] = iron_data + iron_data * 0.11

        cdf["epoch"] = epoch_data
        cdf["epoch_delta"] = epoch_delta

        cdf["DELTA_PLUS_HYDROGEN"] = hydrogen_data * 0.1
        cdf["DELTA_MINUS_HYDROGEN"] = hydrogen_data * 0.1
        cdf["DELTA_PLUS_HELIUM4"] = helium_data * 0.1
        cdf["DELTA_MINUS_HELIUM4"] = helium_data * 0.1
        cdf["DELTA_PLUS_CNO"] = cno_data * 0.1
        cdf["DELTA_MINUS_CNO"] = cno_data * 0.1
        cdf["DELTA_PLUS_NEMGSI"] = nemgsi_data * 0.1
        cdf["DELTA_MINUS_NEMGSI"] = nemgsi_data * 0.1
        cdf["DELTA_PLUS_IRON"] = iron_data * 0.1
        cdf["DELTA_MINUS_IRON"] = iron_data * 0.1

        cdf.new("h_energy", [5, 6, 7], recVary=False)
        cdf.new("h_energy_delta_plus", [0.6, 0.7, 0.6], recVary=False)
        cdf.new("h_energy_delta_minus", [0.4, 0.4, 0.3], recVary=False)

        cdf.new("he4_energy", [5, 6], recVary=False)
        cdf.new("he4_energy_delta_plus", [0.6, 0.7], recVary=False)
        cdf.new("he4_energy_delta_minus", [0.4, 0.4], recVary=False)

        cdf.new("cno_energy", [5, 6], recVary=False)
        cdf.new("cno_energy_delta_plus", [0.6, 0.7], recVary=False)
        cdf.new("cno_energy_delta_minus", [0.4, 0.4], recVary=False)

        cdf.new("nemgsi_energy", [5, 6], recVary=False)
        cdf.new("nemgsi_energy_delta_plus", [0.6, 0.7], recVary=False)
        cdf.new("nemgsi_energy_delta_minus", [0.4, 0.4], recVary=False)

        cdf.new("fe_energy", [5], recVary=False)
        cdf.new("fe_energy_delta_plus", [0.6], recVary=False)
        cdf.new("fe_energy_delta_minus", [0.4], recVary=False)


if __name__ == "__main__":
    path = Path(__file__)
    input_path = path.parent.parent.parent / "instrument_team_data" / "hit" / "hit_l2_sectored_sample1.csv"
    cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "hit" / "imap_hit_l2_sectored-sample1-with-uncertainties_20250101_v001.cdf"

    try:
        create_l2_hit_sectored_rates_cdf_from_csv(str(input_path), str(cdf_file_path))
    except Exception as e:
        traceback.print_exc()
