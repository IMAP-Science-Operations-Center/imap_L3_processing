from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
from astropy.time import Time, TimeDelta
from numpy import ndarray

YEAR_COLUMN = 0
DOY_COLUMN = 1
HOUR_COLUMN = 2
DENSITY_COLUMN_1 = 4
DENSITY_COLUMN_2 = 7
SPEED_COLUMN_1 = 5
SPEED_COLUMN_2 = 8
ALPHA_COL_1 = 6
ALPHA_COL_2 = 9


def validate_omni2_dependency(end_date_exclusive: Time,
                              buffer: TimeDelta, file_path: Path) -> bool:
    omni_data: ndarray = np.loadtxt(file_path, usecols=(0, 1, 2, 5, 23, 24, 27, 30, 31, 34))
    start_year_index = np.searchsorted(omni_data[:, YEAR_COLUMN], (end_date_exclusive + buffer).to_datetime().year)
    omni_data = omni_data[start_year_index:]

    end_year_index = np.searchsorted(omni_data[:, YEAR_COLUMN], (end_date_exclusive + buffer).to_datetime().year,
                                     side='right')
    omni_data = omni_data[:end_year_index]

    start_day_index = np.searchsorted(omni_data[:, DOY_COLUMN],
                                      (end_date_exclusive + buffer).to_datetime().timetuple().tm_yday)
    omni_data = omni_data[start_day_index:]

    end_day_index = np.searchsorted(omni_data[:, DOY_COLUMN],
                                    (end_date_exclusive + buffer).to_datetime().timetuple().tm_yday, side='right')
    omni_data = omni_data[:end_day_index]
    hour_index = np.searchsorted(omni_data[:, HOUR_COLUMN], (end_date_exclusive + buffer).to_datetime().hour)
    omni_data = omni_data[hour_index:]

    # times = Time([f"{int(row[YEAR_COLUMN])}:{int(row[DOY_COLUMN])}:{int(row[HOUR_COLUMN])}:0" for row in omni_data],
    #              format="yday")

    # date_mask = (times >= end_date_exclusive + buffer)
    # omni_data = omni_data[date_mask]

    mask_fill_density_rows = (omni_data[:, DENSITY_COLUMN_1] < 999.9) & (omni_data[:, DENSITY_COLUMN_2] < 999.9)
    mask_fill_speed_rows = (omni_data[:, SPEED_COLUMN_1] < 9999) & (omni_data[:, SPEED_COLUMN_2] < 9999)
    mask_fill_alpha_rows = (omni_data[:, ALPHA_COL_1] < 9.999) & (omni_data[:, ALPHA_COL_2] < 9.999)

    return bool(np.any(mask_fill_density_rows & mask_fill_speed_rows & mask_fill_alpha_rows))


def validate_dependencies(end_date: Time, buffer: TimeDelta, omni2_file_path: Path,
                          f107_index_path: Path, lyman_alpha_path: Path) -> bool:
    omni_condition = validate_omni2_dependency(end_date, buffer, omni2_file_path)

    f107_condition = validate_f107_fluxtable_dependency(end_date, buffer,
                                                        f107_index_path)
    lyman_alpha_condition = validate_lyman_alpha_dependency(end_date, buffer, lyman_alpha_path)

    return omni_condition and f107_condition and lyman_alpha_condition


def validate_f107_fluxtable_dependency(end_date: Time,
                                       buffer: TimeDelta, file_path: Path) -> bool:
    f107_data = np.loadtxt(file_path, dtype=str)
    f107_data = f107_data[2:]
    times = Time([datetime.strptime(row[0], "%Y%m%d") for row in f107_data], format="datetime")

    return times[-1] >= end_date + buffer


def validate_lyman_alpha_dependency(end_date: Time, buffer: TimeDelta, file_path: Path) -> bool:
    cdf = netCDF4.Dataset(file_path)

    return cdf.time_coverage_end >= end_date + buffer
