from pathlib import Path

import numpy as np
from astropy.time import Time

from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import jd_fm_Carrington


def filter_l3a_files(l3a_data: list[dict], bad_day_list_path: Path, cr: int) -> list[dict]:
    carrington_start_date = jd_fm_Carrington(float(cr))
    cr_start_time = Time(carrington_start_date, format='jd')
    cr_start_time.format = 'iso'
    carrington_end_date_non_inclusive = jd_fm_Carrington(cr + 1)
    cr_end_time = Time(carrington_end_date_non_inclusive, format='jd')
    cr_end_time.format = 'iso'

    def mean_time(l3a_dict):
        time = Time(l3a_dict['start_time']) + 0.5 * (Time(l3a_dict['end_time']) - Time(l3a_dict['start_time']))
        return time

    l3a_in_cr = [l3a_dict for l3a_dict in l3a_data if cr_start_time <= mean_time(l3a_dict) < cr_end_time]

    with open(bad_day_list_path, 'r') as bad_day_file:
        bad_day_list = Time(np.genfromtxt(bad_day_file, dtype=str)[:, 1])
        sorted_bad_days = bad_day_list.sort()
        filtered_data = []
        l3a_data_index = 0
        bad_day_list_index = 0
        while l3a_data_index < len(l3a_in_cr) and bad_day_list_index < len(sorted_bad_days):
            start_time = Time(l3a_in_cr[l3a_data_index]['start_time'])
            end_time = Time(l3a_in_cr[l3a_data_index]['end_time'])
            if start_time <= sorted_bad_days[bad_day_list_index] < end_time:
                l3a_data_index += 1
            elif sorted_bad_days[bad_day_list_index] > end_time:
                filtered_data.append(l3a_in_cr[l3a_data_index])
                l3a_data_index += 1
            else:
                bad_day_list_index += 1

        if l3a_data_index < len(l3a_in_cr):
            filtered_data += l3a_in_cr[l3a_data_index:]

    return filtered_data
