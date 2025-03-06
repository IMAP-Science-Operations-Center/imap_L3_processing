from datetime import datetime, timedelta

import numpy as np

from imap_processing.constants import ONE_SECOND_IN_NANOSECONDS


def rebin(from_epoch: np.ndarray[float], from_data: np.ndarray[float], to_epoch: np.ndarray[float],
          to_epoch_delta: np.ndarray[timedelta]) -> np.ndarray[float]:
    output_shape = to_epoch.shape + from_data.shape[1:]
    vector_sums = np.zeros(shape=output_shape, dtype=float)
    vector_counts = np.zeros(shape=output_shape, dtype=float)

    input_data_iter = ((time, vec) for time, vec in zip(from_epoch, from_data))
    current_epoch, current_vec = next(input_data_iter, (None, None))
    for i, (time, delta) in enumerate(zip(to_epoch, to_epoch_delta)):
        start_time = time - delta.total_seconds() * ONE_SECOND_IN_NANOSECONDS
        end_time = time + delta.total_seconds() * ONE_SECOND_IN_NANOSECONDS
        print(current_epoch)
        print(start_time)
        print(end_time)
        print(ONE_SECOND_IN_NANOSECONDS)

        while current_epoch is not None and current_epoch < start_time:
            current_epoch, current_vec = next(input_data_iter, (None, None))

        while current_epoch is not None and start_time <= current_epoch < end_time:
            vector_sums[i] += current_vec
            vector_counts[i] += 1
            current_epoch, current_vec = next(input_data_iter, (None, None))

    return np.divide(vector_sums, vector_counts, out=np.full_like(vector_sums, fill_value=np.nan),
                     where=vector_counts != 0)


def find_closest_neighbor(from_epoch: np.ndarray[datetime], from_data: np.ndarray[float],
                          to_epoch: np.ndarray[datetime],
                          maximum_distance: timedelta) -> np.ndarray[float]:
    from_epoch_as_dt64 = from_epoch.astype(np.datetime64)
    to_epoch_as_dt64 = to_epoch.astype(np.datetime64)
    index = np.searchsorted(from_epoch_as_dt64, to_epoch_as_dt64)
    right_indices = np.minimum(len(from_epoch_as_dt64) - 1, index)
    left_indices = np.maximum(0, index - 1)

    right_delta = np.abs(from_epoch_as_dt64[right_indices] - to_epoch_as_dt64)
    left_delta = np.abs(from_epoch_as_dt64[left_indices] - to_epoch_as_dt64)

    best_indices = np.where(right_delta < left_delta, right_indices, left_indices)

    min_deltas = np.abs(from_epoch_as_dt64[best_indices] - to_epoch_as_dt64)
    min_outside_range = min_deltas > maximum_distance

    closest_data = from_data[best_indices].astype(float, copy=True)
    closest_data[min_outside_range] = np.nan

    return closest_data
