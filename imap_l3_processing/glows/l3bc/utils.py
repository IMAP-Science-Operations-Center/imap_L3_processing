from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from imap_processing.spice.repoint import get_repoint_data


def get_best_ancillary(
    start_date: datetime, end_date: datetime, ancillary_query_results: list[dict]
) -> str | None:
    valid_ancillaries = []
    for ancillary_file in ancillary_query_results:
        ancillary_start_date = datetime.strptime(ancillary_file["start_date"], "%Y%m%d")
        ancillary_end_date = (
            datetime.strptime(ancillary_file["end_date"], "%Y%m%d")
            if ancillary_file["end_date"]
            else None
        )

        if ancillary_start_date <= end_date and (
            ancillary_end_date is None or ancillary_end_date >= start_date
        ):
            valid_ancillaries.append(ancillary_file)

    if len(valid_ancillaries) == 0:
        return None
    else:
        latest_ancillary = max(valid_ancillaries, key=lambda x: x["ingestion_date"])
        return Path(latest_ancillary["file_path"]).name


carrington_first = 1
first_carrington_start_date = datetime(1853, 11, 9, 19, 53, 45, 600000)
carrington_length = timedelta(days=27.2753)


def get_date_range_of_cr(cr_number: int) -> tuple[datetime, datetime]:
    start_date = (
        first_carrington_start_date + (cr_number - carrington_first) * carrington_length
    )
    return start_date, start_date + carrington_length


def get_midpoint_of_cr(cr_number: int) -> datetime:
    start, _ = get_date_range_of_cr(cr_number)
    return start + carrington_length / 2


def get_cr_for_date_time(datetime_to_check: datetime) -> int:
    return int(
        carrington_first
        + (datetime_to_check - first_carrington_start_date) / carrington_length
    )


def get_pointing_date_range(repointing: int) -> (datetime, datetime):
    repointing_df: pd.DataFrame = get_repoint_data()
    matching_rows_start = repointing_df[repointing_df["repoint_id"] == repointing]
    matching_rows_end = repointing_df[repointing_df["repoint_id"] == repointing + 1]
    if len(matching_rows_start) == 0 or len(matching_rows_end) == 0:
        raise ValueError(f"No pointing found for pointing: {repointing}")
    repointing_data_start = matching_rows_start.iloc[0]
    repointing_data_end = matching_rows_end.iloc[0]
    start_time = repointing_data_start["repoint_end_utc"]
    end_time = repointing_data_end["repoint_start_utc"]

    return np.datetime64(start_time).astype(datetime), np.datetime64(end_time).astype(
        datetime
    )
