import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)


def calculate_coincidence_rate(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
) -> NDArray:
    speed_in_sw_frame = chunk_response.speed_in_sw_frame
    bin_weights = chunk_response.bin_weights
    cutoff_speed = fitting_params.cutoff_speed

    f_pui = vasyliunas_siscoe_distribution.f(speed_in_sw_frame, fitting_params)
    apply_partial_heaviside_at_cutoff(f_pui, speed_in_sw_frame, cutoff_speed)

    rates = np.tensordot(bin_weights, f_pui, axes=([2], [0]))
    rates += fitting_params.background_count_rate
    return rates


def apply_partial_heaviside_at_cutoff(
    f_pui: NDArray, speed_in_sw_frame: NDArray, cutoff_speed: float
) -> None:
    delta_v_prime = speed_in_sw_frame[1] - speed_in_sw_frame[0]
    cutoff_index = int((cutoff_speed - speed_in_sw_frame[0]) / delta_v_prime)
    if 0 <= cutoff_index < speed_in_sw_frame.size - 1:
        cell_left_edge = speed_in_sw_frame[cutoff_index] - delta_v_prime / 2
        f_pui[cutoff_index] *= (cutoff_speed - cell_left_edge) / delta_v_prime
