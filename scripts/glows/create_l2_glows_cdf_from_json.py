import json
import math
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TTJ2000_EPOCH
from tests.test_helpers import get_test_data_path


def fill_official_l2_cdf_with_json_values(output_folder: Path, input_path: Path) -> Path:
    official_l2_path = get_test_data_path("glows/imap_glows_l2_hist_20130908-repoint01000_v001.cdf")

    with open(input_path) as f:
        instrument_data = json.load(f)

        start_of_epoch_window = datetime.fromisoformat(instrument_data["start_time"]) + timedelta(days=5479)

        end_of_epoch_window = datetime.fromisoformat(instrument_data["end_time"]) + timedelta(days=5479)
        epoch_window = end_of_epoch_window - start_of_epoch_window
        epoch = start_of_epoch_window + epoch_window / 2

        repoint_id = 896 + math.floor((epoch - datetime(2025, 1, 1)) / timedelta(days=1))

        new_name = ScienceFilePath.generate_from_inputs(
            instrument="glows",
            data_level="l2",
            descriptor="hist",
            start_time=start_of_epoch_window.strftime("%Y%m%d"),
            version="v003",
            extension="cdf",
            repointing=repoint_id
        ).filename

        new_file_path = output_folder / new_name
        new_file_path.unlink(missing_ok=True)

        with CDF(str(new_file_path), masterpath=str(official_l2_path)) as cdf:
            print(f"Writing to file {new_name}")
            cdf["epoch"][0] = epoch
            cdf['start_time'][0] = (start_of_epoch_window - TTJ2000_EPOCH).total_seconds() * 1e9
            cdf['end_time'][0] = (end_of_epoch_window - TTJ2000_EPOCH).total_seconds() * 1e9

            lightcurve_vars = [
                "spin_angle",
                "photon_flux",
                "exposure_times",
                "flux_uncertainties",
                "ecliptic_lon",
                "ecliptic_lat",
            ]
            for var in lightcurve_vars:
                cdf[var] = np.array(instrument_data["daily_lightcurve"][var])[np.newaxis, :]

            cdf["raw_histograms"] = np.array(instrument_data["daily_lightcurve"]["raw_histogram"])[np.newaxis, :]
            cdf["histogram_flag_array"] = np.array(
                [int(f, 16) for f in instrument_data["daily_lightcurve"]["histogram_flag_array"]])[np.newaxis, :]

            single_value_vars = [
                "filter_temperature_average",
                "filter_temperature_std_dev",
                "hv_voltage_average",
                "hv_voltage_std_dev",
                "spin_period_average",
                "spin_period_std_dev",
                "pulse_length_average",
                "pulse_length_std_dev",
                "spin_period_ground_average",
                "spin_period_ground_std_dev",
                "position_angle_offset_average",
                "position_angle_offset_std_dev",
                "identifier"
            ]
            for var in single_value_vars:
                cdf[var][0] = instrument_data[var]

            vector_vars = [
                "spacecraft_location_average",
                "spacecraft_location_std_dev",
                "spacecraft_velocity_average",
                "spacecraft_velocity_std_dev",
                "spin_axis_orientation_average",
                "spin_axis_orientation_std_dev"
            ]
            for var in vector_vars:
                cdf[var][0] = np.array(list(instrument_data[var].values()))

            cdf["bad_time_flag_occurrences"][0] = list(
                instrument_data["bad_time_flag_occurences"].values())  # for old instrument team data
            cdf["number_of_good_l1b_inputs"][0] = instrument_data["header"]["number_of_l1b_files_used"]
            cdf["total_l1b_inputs"][0] = instrument_data["header"]["number_of_all_l1b_files"]


if __name__ == "__main__":
    json_directory = Path(
        r'/Users/harrison/Downloads/data_products_cbk_implementation_2024-12-31_1year/data_l2_histograms')
    output_directory = Path(r'/Users/harrison/Downloads/l2_cdfs_pre_timeshifted')
    shutil.rmtree(output_directory, ignore_errors=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    for file_path in json_directory.iterdir():
        try:
            fill_official_l2_cdf_with_json_values(output_directory, file_path)
        except Exception as e:
            traceback.print_exc()
