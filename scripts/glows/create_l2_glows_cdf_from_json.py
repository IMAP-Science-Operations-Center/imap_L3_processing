import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF


def set_up_required_data_values(cdf, l2_data):
    for key in ['identifier', 'filter_temperature_average', 'filter_temperature_std_dev',
                'hv_voltage_average', 'hv_voltage_std_dev', 'spin_period_average', 'spin_period_std_dev',
                'spin_period_ground_average', 'spin_period_ground_std_dev', 'pulse_length_average',
                'pulse_length_std_dev',
                'position_angle_offset_average', 'position_angle_offset_std_dev']:
        cdf[key] = [l2_data[key]]

    for key1 in ['spin_axis_orientation_average', 'spin_axis_orientation_std_dev']:
        for key2 in ['lon', 'lat']:
            cdf[f"{key1}_{key2}"] = [l2_data[key1][key2]]

    for key1 in ['spacecraft_location_average', 'spacecraft_location_std_dev',
                 'spacecraft_velocity_average', 'spacecraft_velocity_std_dev']:
        for key2 in ['x', 'y', 'z']:
            cdf[f"{key1}_{key2}"] = [l2_data[key1][key2]]


def create_l2_glows_cdf_from_json(json_file_path: str, output_filename: str):
    with CDF(output_filename, '') as cdf:
        cdf.col_major(True)
        with open(json_file_path) as f:
            data = json.load(f)
            set_up_required_data_values(cdf, data)
            light_curve = data["daily_lightcurve"]

            start_of_epoch_window = datetime.fromisoformat(data["start_time"])
            end_of_epoch_window = datetime.fromisoformat(data["end_time"])
            epoch_window = end_of_epoch_window - start_of_epoch_window
            epoch = start_of_epoch_window + epoch_window / 2

            cdf["start_time"] = np.array([start_of_epoch_window])
            cdf["end_time"] = np.array([end_of_epoch_window])
            cdf["epoch"] = np.array([epoch])
            cdf["epoch_delta"] = np.array([int(epoch_window.total_seconds() * 1e9 / 2)])
            int_flags = [int(f, 16) for f in light_curve["histogram_flag_array"]]
            flag_bits = np.unpackbits(np.array([int_flags], dtype=np.uint8), axis=0, bitorder='little', count=4)
            cdf["histogram_flag_array"] = [flag_bits]

            cdf.new("number_of_bins", light_curve["number_of_bins"], recVary=False)

            lightcurve_vars = [
                "spin_angle", "photon_flux", "exposure_times", "flux_uncertainties",
                "raw_histogram", "ecliptic_lon", "ecliptic_lat"
            ]
            for var in lightcurve_vars:
                cdf[var] = np.reshape(light_curve[var], (1, -1))

            cdf.attrs["flight_software_version"] = data["header"]["flight_software_version"]
            cdf.attrs["pkts_file_name"] = data["header"]["pkts_file_name"]
            cdf.attrs["ancillary_data_files"] = data["header"]["ancillary_data_files"]


if __name__ == "__main__":
    json_directory = Path(r'C:\Users\Harrison\Downloads\data_l2_histograms')
    output_directory = Path(r'C:\Users\Harrison\Downloads\p3')
    for file_path in json_directory.iterdir():
        output = re.search(r"imap_glows_l2_(\d{8}).*_orbX_modX_p_v00.json", file_path.name)
        output_file_path_date = output.group(1)
        cdf_file_path = f"{output_directory}/imap_glows_l2_hist_{output_file_path_date}_v003.cdf"
        try:
            create_l2_glows_cdf_from_json(file_path, cdf_file_path)
        except Exception as e:
            traceback.print_exc()
