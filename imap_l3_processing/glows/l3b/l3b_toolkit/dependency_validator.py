from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
from astropy.time import Time
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


def validate_omni2_dependency(start_date_inclusive: Time,
                              end_date_exclusive: Time, file_path: Path) -> bool:
    omni_data: ndarray = np.loadtxt(file_path, usecols=(0, 1, 2, 5, 23, 24, 27, 30, 31, 34))

    times = Time([f"{int(row[YEAR_COLUMN])}:{int(row[DOY_COLUMN])}:{int(row[HOUR_COLUMN])}:0" for row in omni_data],
                 format="yday")

    if times[-1] < end_date_exclusive:
        return False

    date_mask = (start_date_inclusive <= times) & (end_date_exclusive > times)
    omni_data = omni_data[date_mask]

    mask_fill_density_rows = (omni_data[:, DENSITY_COLUMN_1] < 999.9) & (omni_data[:, DENSITY_COLUMN_2] < 999.9)
    mask_fill_speed_rows = (omni_data[:, SPEED_COLUMN_1] < 9999) & (omni_data[:, SPEED_COLUMN_2] < 9999)
    mask_fill_alpha_rows = (omni_data[:, ALPHA_COL_1] < 9.999) & (omni_data[:, ALPHA_COL_2] < 9.999)

    return bool(np.all(mask_fill_density_rows & mask_fill_speed_rows & mask_fill_alpha_rows))


def validate_dependencies(start_date_inclusive: Time, end_date_exclusive: Time, omni2_file_path: Path,
                          fluxtable_file_path: Path, lyman_alpha_path: Path) -> bool:
    omni_condition = validate_omni2_dependency(start_date_inclusive, end_date_exclusive, omni2_file_path)
    f107_condition = validate_f107_fluxtable_dependency(start_date_inclusive, end_date_exclusive,
                                                        fluxtable_file_path)
    lyman_alpha_condition = validate_lyman_alpha_dependency(end_date_exclusive, lyman_alpha_path)

    return omni_condition and f107_condition and lyman_alpha_condition


def validate_f107_fluxtable_dependency(start_date_inclusive: Time,
                                       end_date_exclusive: Time, file_path: Path) -> bool:
    f107_data = np.loadtxt(file_path, dtype=str)

    times = Time([datetime.strptime(row[0], "%Y%m%d") for row in f107_data], format="datetime")

    return end_date_exclusive < times[-1] and start_date_inclusive > times[0]


def validate_lyman_alpha_dependency(end_date_exclusive: Time, file_path: Path) -> bool:
    cdf = netCDF4.Dataset(file_path)

    return end_date_exclusive < cdf.time_coverage_end
