import os
import sys
from pathlib import Path

from spacepy.pycdf import CDF

import imap_processing
from swapi.swapi_alpha_sw_speed_temperature_density_demo import read_l2_data_from_dat


def generate_cdf_from_dat(dat_file_path: str, output_filename: str):
    dat_data = read_l2_data_from_dat(dat_file_path)
    cdf_file_path = Path(imap_processing.__file__).parent.parent / 'swapi' / 'test_data' / output_filename

    with CDF(str(cdf_file_path), '') as cdf:
        cdf.col_major(True)

        cdf.new("energy", dat_data.energy, recVary=False)
        cdf["epoch"] = dat_data.epoch
        cdf["spin_angles"] = dat_data.spin_angles
        cdf["swp_coin_rate"] = dat_data.coincidence_count_rate
        cdf["swp_coin_unc"] = dat_data.coincidence_count_rate_uncertainty


if __name__ == "__main__":
    generate_cdf_from_dat(os.path.abspath("test_data/swapi_test_data_50_sweeps.dat"),
                          "imap_swapi_l2_50-sweeps_20100101_v001")
