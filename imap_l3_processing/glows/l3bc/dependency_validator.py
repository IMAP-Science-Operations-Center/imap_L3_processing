from datetime import datetime, timedelta
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


def validate_omni2_dependency(cr_start_date: datetime,
                              cr_end_date_exclusive: datetime, file_path: Path) -> bool:
    omni_data: ndarray = np.loadtxt(file_path, usecols=(0, 1, 2, 5, 23, 24, 27, 30, 31, 34))

    row_day_keys = omni_data[:, YEAR_COLUMN] * 1000 + omni_data[:, DOY_COLUMN]
    start_key = cr_start_date.year * 1000 + cr_start_date.timetuple().tm_yday
    end_key = cr_end_date_exclusive.year * 1000 + cr_end_date_exclusive.timetuple().tm_yday

    start_idx = np.searchsorted(row_day_keys, start_key)
    end_idx = np.searchsorted(row_day_keys, end_key)
    omni_data = omni_data[start_idx:end_idx]

    for data_point in omni_data:
        mask_fill_density_rows = np.all(data_point[[DENSITY_COLUMN_1, DENSITY_COLUMN_2]] < 999.9)
        mask_fill_speed_rows = np.all(data_point[[SPEED_COLUMN_1, SPEED_COLUMN_2]] < 9999)

        if np.all([mask_fill_density_rows, mask_fill_speed_rows]):
            return True

    return False

def validate_dependencies(cr_start_date: datetime, cr_end_date: datetime, buffer: timedelta,
                          omni2_file_path: Path,
                          f107_index_path: Path, lyman_alpha_path: Path) -> bool:
    omni_condition = validate_omni2_dependency(cr_start_date, cr_end_date, omni2_file_path)

    f107_condition = validate_f107_fluxtable_dependency(cr_end_date, buffer,
                                                        f107_index_path)
    lyman_alpha_condition = validate_lyman_alpha_dependency(cr_end_date, buffer, lyman_alpha_path)

    return omni_condition and f107_condition and lyman_alpha_condition


def validate_f107_fluxtable_dependency(end_date: datetime,
                                       buffer: timedelta, file_path: Path) -> bool:
    f107_data = np.loadtxt(file_path, dtype=str)
    f107_data = f107_data[2:]
    last_date = f107_data[-1][0]

    return datetime.strptime(last_date, "%Y%m%d") >= end_date + buffer


def validate_lyman_alpha_dependency(end_date: datetime, buffer: timedelta, file_path: Path) -> bool:
    cdf = netCDF4.Dataset(file_path)

    isoformat_time = cdf.time_coverage_end.replace('Z', '+00:00')

    datetime_from_iso = datetime.fromisoformat(isoformat_time).replace(tzinfo=None)

    return datetime_from_iso >= end_date + buffer
