from dataclasses import dataclass

import numpy as np

from imap_l3_processing.hit.l3.electron.science.electron_intensity_factors import (
    ElectronIntensityFactors,
)

SECONDS_PER_L1B_FRAME = 60
IALIRT_RATE_COUNT = 20


@dataclass
class ElectronIntensities:
    a_side: np.ndarray
    a_side_delta_plus: np.ndarray
    a_side_delta_minus: np.ndarray
    b_side: np.ndarray
    b_side_delta_plus: np.ndarray
    b_side_delta_minus: np.ndarray


def calculate_electron_intensities(
    rates: np.ndarray,
    rates_delta_plus: np.ndarray,
    rates_delta_minus: np.ndarray,
    factors: ElectronIntensityFactors,
) -> ElectronIntensities:
    if rates.shape[1] != IALIRT_RATE_COUNT:
        raise ValueError(
            f"Expected {IALIRT_RATE_COUNT} I-ALiRT rates per epoch, got {rates.shape[1]}"
        )
    per_bin_divisor = SECONDS_PER_L1B_FRAME * factors.delta_e_mev * factors.geometry_factor_cm2_sr
    return ElectronIntensities(
        a_side=(rates[:, factors.a_side_rate_indices] - factors.background) / per_bin_divisor,
        a_side_delta_plus=rates_delta_plus[:, factors.a_side_rate_indices] / per_bin_divisor,
        a_side_delta_minus=rates_delta_minus[:, factors.a_side_rate_indices] / per_bin_divisor,
        b_side=(rates[:, factors.b_side_rate_indices] - factors.background) / per_bin_divisor,
        b_side_delta_plus=rates_delta_plus[:, factors.b_side_rate_indices] / per_bin_divisor,
        b_side_delta_minus=rates_delta_minus[:, factors.b_side_rate_indices] / per_bin_divisor,
    )
