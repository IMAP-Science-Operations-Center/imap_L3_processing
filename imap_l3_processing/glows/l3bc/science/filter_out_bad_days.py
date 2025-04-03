from pathlib import Path

import numpy as np
from astropy.time import Time


def filter_out_bad_days(l3a_data: [dict], bad_day_list_path: Path) -> [dict]:
    with open(bad_day_list_path, 'r') as bad_day_file:
        bad_day_list = Time(np.genfromtxt(bad_day_file, dtype=str)[:, 1])
        sorted_bad_days = bad_day_list.sort()
        filtered_data = []
        l3a_data_index = 0
        bad_day_list_index = 0
        while l3a_data_index < len(l3a_data) and bad_day_list_index < len(sorted_bad_days):
            start_time = Time(l3a_data[l3a_data_index]['start_time'])
            end_time = Time(l3a_data[l3a_data_index]['end_time'])
            if start_time <= sorted_bad_days[bad_day_list_index] < end_time:
                l3a_data_index += 1
            elif sorted_bad_days[bad_day_list_index] > end_time:
                filtered_data.append(l3a_data[l3a_data_index])
                l3a_data_index += 1
            else:
                bad_day_list_index += 1

        if l3a_data_index < len(l3a_data):
            filtered_data += l3a_data[l3a_data_index:]

    return filtered_data
