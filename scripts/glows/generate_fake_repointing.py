from typing import Optional, Union

import numpy as np
import pandas as pd

TTJ2000_EPOCH = np.datetime64("2010-01-01T00:00:00.000", "ns")


def generate_repoint_data(
        repoint_start_met: Union[float, np.ndarray],
        repoint_end_met: Optional[Union[float, np.ndarray]] = None,
        repoint_id_start: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Generate a repoint dataframe for the star/end times provided.

    Parameters
    ----------
    repoint_start_met : float, np.ndarray
            Provides the repoint start time(s) in Mission Elapsed Time (MET).
    repoint_end_met : float, np.ndarray, optional
        Provides the repoint end time(s) in MET. If not provided, end times
        will be 15 minutes after start times.
    repoint_id_start : int, optional
        Provides the starting repoint id number of the first repoint in the
        generated data.

    Returns
    -------
    repoint_df : pd.DataFrame
        Repoint dataframe with start and end repoint times provided and incrementing
        repoint_ids starting at 1.
    """
    repoint_start_times = np.array(repoint_start_met)
    if repoint_end_met is None:
        repoint_end_met = repoint_start_times + 15 * 60
    # Calculate UTC times without spice (accepting ~5 second inaccuracy)
    repoint_start_dt64 = TTJ2000_EPOCH + (repoint_start_times * 1e9).astype(
        "timedelta64[ns]"
    )
    repoint_end_dt64 = TTJ2000_EPOCH + (repoint_end_met * 1e9).astype("timedelta64[ns]")
    repoint_df = pd.DataFrame.from_dict(
        {
            "repoint_start_sec_sclk": repoint_start_times.astype(int),
            "repoint_start_subsec_sclk": ((repoint_start_times % 1.0) * 1e6).astype(
                int
            ),
            "repoint_start_utc": np.datetime_as_string(repoint_start_dt64, unit="us"),
            "repoint_end_sec_sclk": repoint_end_met.astype(int),
            "repoint_end_subsec_sclk": ((repoint_end_met % 1.0) * 1e6).astype(int),
            "repoint_end_utc": np.datetime_as_string(repoint_end_dt64, unit="us"),
            "repoint_id": np.arange(repoint_start_times.size, dtype=int)
                          + repoint_id_start,
        }
    )
    return repoint_df


start_time = 473385603

generate_repoint_data(np.arange(473385603, 504921603, 86400)).to_csv('2025_fake_repointing.csv')
