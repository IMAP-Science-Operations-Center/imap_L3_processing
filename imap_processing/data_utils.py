from datetime import datetime, timedelta

import numpy as np


def rebin(from_epoch: np.ndarray[float], from_data: np.ndarray[float], to_epoch: np.ndarray[float],
          to_epoch_delta: np.ndarray[float]) -> np.ndarray[float]:
    output_shape = to_epoch.shape + from_data.shape[1:]
    vector_sums = np.zeros(shape=output_shape, dtype=float)
    vector_counts = np.zeros(shape=output_shape, dtype=float)

    input_data_iter = ((time, vec) for time, vec in zip(from_epoch, from_data))
    current_epoch, current_vec = next(input_data_iter, (None, None))
    for i, (time, delta) in enumerate(zip(to_epoch, to_epoch_delta)):
        start_time = time - delta
        end_time = time + delta

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
    deltas = np.abs(from_epoch - to_epoch[..., np.newaxis])
    min_deltas = np.min(deltas, axis=-1, keepdims=True)
    min_outside_range = min_deltas > maximum_distance
    indices = np.argmin(deltas, axis=-1)

    closest_data = from_data[indices]
    closest_data_in_range = np.where(min_outside_range, np.nan, closest_data)
    return closest_data_in_range
