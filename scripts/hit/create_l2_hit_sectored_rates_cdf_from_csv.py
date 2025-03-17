import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS


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
        cdf["h"] = hydrogen_data
        cdf["delta_minus_h"] = hydrogen_data - hydrogen_data * 0.09
        cdf["delta_plus_h"] = hydrogen_data + hydrogen_data * 0.11
        cdf["he4"] = helium_data
        cdf["delta_minus_he4"] = helium_data - helium_data * 0.09
        cdf["delta_plus_he4"] = helium_data + helium_data * 0.11
        cdf["cno"] = cno_data
        cdf["delta_minus_cno"] = cno_data - cno_data * 0.09
        cdf["delta_plus_cno"] = cno_data + cno_data * 0.11
        cdf["nemgsi"] = nemgsi_data
        cdf["delta_minus_nemgsi"] = nemgsi_data - nemgsi_data * 0.09
        cdf["delta_plus_nemgsi"] = nemgsi_data + nemgsi_data * 0.11
        cdf["fe"] = iron_data
        cdf["delta_minus_fe"] = iron_data - iron_data * 0.09
        cdf["delta_plus_fe"] = iron_data + iron_data * 0.11

        cdf["epoch"] = epoch_data
        cdf["epoch_delta"] = epoch_delta

        cdf["delta_plus_h"] = hydrogen_data * 0.1
        cdf["delta_minus_h"] = hydrogen_data * 0.1
        cdf["delta_plus_he4"] = helium_data * 0.1
        cdf["delta_minus_he4"] = helium_data * 0.1
        cdf["delta_plus_cno"] = cno_data * 0.1
        cdf["delta_minus_cno"] = cno_data * 0.1
        cdf["delta_plus_nemgsi"] = nemgsi_data * 0.1
        cdf["delta_minus_nemgsi"] = nemgsi_data * 0.1
        cdf["delta_plus_fe"] = iron_data * 0.1
        cdf["delta_minus_fe"] = iron_data * 0.1

        # using energy values listed in Table 3 of algorithm document
        cdf.new("h_energy_mean", [2.7, 5, 8], recVary=False)
        cdf.new("h_energy_delta_plus", [0.9, 1, 2], recVary=False)
        cdf.new("h_energy_delta_minus", [0.9, 1, 2], recVary=False)

        cdf.new("he4_energy_mean", [5, 9], recVary=False)
        cdf.new("he4_energy_delta_plus", [1, 3], recVary=False)
        cdf.new("he4_energy_delta_minus", [1, 3], recVary=False)

        cdf.new("cno_energy_mean", [5, 9], recVary=False)
        cdf.new("cno_energy_delta_plus", [1, 3], recVary=False)
        cdf.new("cno_energy_delta_minus", [1, 3], recVary=False)

        cdf.new("nemgsi_energy_mean", [5, 9], recVary=False)
        cdf.new("nemgsi_energy_delta_plus", [1, 3], recVary=False)
        cdf.new("nemgsi_energy_delta_minus", [1, 3], recVary=False)

        cdf.new("fe_energy_mean", [8], recVary=False)
        cdf.new("fe_energy_delta_plus", [4], recVary=False)
        cdf.new("fe_energy_delta_minus", [4], recVary=False)


if __name__ == "__main__":
    path = Path(__file__)
    input_path = path.parent.parent.parent / "instrument_team_data" / "hit" / "hit_l2_sectored_sample1.csv"
    cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "hit" / "imap_hit_l2_macropixel-intensity_20250101_v002.cdf"

    try:
        create_l2_hit_sectored_rates_cdf_from_csv(str(input_path), str(cdf_file_path))
    except Exception as e:
        traceback.print_exc()
