from __future__ import annotations

import numpy as np
import uncertainties
from numpy.typing import NDArray

from imap_l3_processing.constants import (
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN,
    CENTIMETERS_PER_METER,
    HE_PUI_PARTICLE_MASS_KG,
    METERS_PER_KILOMETER,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    apply_partial_heaviside_at_cutoff,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)


def calculate_helium_pui_density(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
) -> float:
    speed_in_sw_frame = chunk_response.speed_in_sw_frame
    integration_weights = _moment_integration_weights(speed_in_sw_frame)

    @uncertainties.wrap
    def calculate(
        cooling_index: float,
        ionization_rate: float,
        cutoff_speed: float,
        background_count_rate: float,
    ):
        params = FittingParameters(
            cooling_index, ionization_rate, cutoff_speed, background_count_rate
        )
        f_pui = np.asarray(
            vasyliunas_siscoe_distribution.f(speed_in_sw_frame, params), dtype=float
        ).copy()
        apply_partial_heaviside_at_cutoff(f_pui, speed_in_sw_frame, cutoff_speed)
        integral = float(np.sum(integration_weights * f_pui))
        return (
            4 * np.pi * integral / (CENTIMETERS_PER_METER * METERS_PER_KILOMETER) ** 3
        )

    return calculate(
        fitting_params.cooling_index,
        fitting_params.ionization_rate,
        fitting_params.cutoff_speed,
        fitting_params.background_count_rate,
    )


def calculate_helium_pui_temperature(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    fitting_params: FittingParameters,
) -> float:
    speed_in_sw_frame = chunk_response.speed_in_sw_frame
    integration_weights = _moment_integration_weights(speed_in_sw_frame)

    @uncertainties.wrap
    def calculate(
        cooling_index: float,
        ionization_rate: float,
        cutoff_speed: float,
        background_count_rate: float,
    ):
        params = FittingParameters(
            cooling_index, ionization_rate, cutoff_speed, background_count_rate
        )
        f_pui = np.asarray(
            vasyliunas_siscoe_distribution.f(speed_in_sw_frame, params), dtype=float
        ).copy()
        apply_partial_heaviside_at_cutoff(f_pui, speed_in_sw_frame, cutoff_speed)
        weighted = integration_weights * f_pui
        denominator = float(np.sum(weighted))
        numerator = float(np.sum(speed_in_sw_frame * speed_in_sw_frame * weighted))
        return (
            HE_PUI_PARTICLE_MASS_KG
            / (3 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
            * numerator
            / denominator
            * METERS_PER_KILOMETER**2
        )

    return calculate(
        fitting_params.cooling_index,
        fitting_params.ionization_rate,
        fitting_params.cutoff_speed,
        fitting_params.background_count_rate,
    )


def _moment_integration_weights(speed_in_sw_frame: NDArray) -> NDArray:
    delta_v_prime = speed_in_sw_frame[1] - speed_in_sw_frame[0]
    trapz_weights = np.full(speed_in_sw_frame.shape, delta_v_prime)
    trapz_weights[0] *= 0.5
    trapz_weights[-1] *= 0.5
    return speed_in_sw_frame**2 * trapz_weights
