import os

from spacepy.pycdf import CDF
from spacepy.pycdf.const import CDF_TIME_TT2000

from scripts.swapi.swapi_alpha_sw_speed_temperature_density_demo import read_l2_data_from_dat
from tests.test_helpers import get_test_data_path


def generate_cdf_from_dat(dat_file_path: str, output_filename: str):
    dat_data = read_l2_data_from_dat(dat_file_path)
    cdf_file_path = get_test_data_path('swapi') / output_filename

    with CDF(str(cdf_file_path), '') as cdf:
        cdf.col_major(True)

        cdf.new("swp_esa_energy", dat_data.energy, recVary=False)
        cdf["swp_esa_energy"].attrs["FILLVAL"] = -1e31
        cdf.new("sci_start_time", dat_data.sci_start_time, type=CDF_TIME_TT2000)
        cdf["spin_angles"] = dat_data.spin_angles
        cdf["spin_angles"].attrs["FILLVAL"] = -1e31
        cdf["swp_coin_rate"] = dat_data.coincidence_count_rate
        cdf["swp_coin_rate"].attrs["FILLVAL"] = -1e31

        cdf["swp_coin_rate_stat_uncert_plus"] = dat_data.coincidence_count_rate_uncertainty
        cdf["swp_coin_rate_stat_uncert_plus"].attrs["FILLVAL"] = -1e31



if __name__ == "__main__":
    generate_cdf_from_dat(os.path.abspath("../../instrument_team_data/swapi/swapi_test_data_sept_2025.dat"),
                          "imap_swapi_l2_sci_20100101_v001")
