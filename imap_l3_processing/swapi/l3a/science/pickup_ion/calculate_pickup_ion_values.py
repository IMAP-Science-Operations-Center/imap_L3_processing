from __future__ import annotations

from dataclasses import dataclass

import lmfit
import numdifftools as ndt
import numpy as np
from imap_processing.swapi.l2 import swapi_l2
from lmfit import Parameters
from numpy import ndarray
from scipy.linalg import inv
from uncertainties import ufloat, nominal_value
from dataclasses import astuple
from imap_l3_processing.constants import ONE_AU_IN_KM
from imap_l3_processing.swapi.constants import (
    SWAPI_BACKGROUND_RATE,
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.calculate_coincidence_rate import (
    calculate_coincidence_rate,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.collapsed_response_grid import (
    ChunkCollapsedResponse,
    build_chunk_collapsed_response,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.goodness_of_fit import is_good_fit
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    FittingParameters,
    VasyliunasSiscoeDistribution,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.species import Species

_SWEEPS_PER_CHUNK = 50
_SWEEP_LEN = 72
_COARSE_SWEEP_LEN = 62
_PICKUP_ION_SPECIES = Species.HELIUM_PLUS


@dataclass
class PickupIonFitResult:
    fitting_params: FittingParameters
    chunk_response: ChunkCollapsedResponse
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution


def calculate_pickup_ion_values(
    swapi_response: SwapiResponse,
    esa_energies: np.ndarray,
    count_rates: np.ndarray,
    bulk_sw_per_bin_swapi_kms: ndarray,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    time_as_tt2000: int,
) -> PickupIonFitResult:
    """
    Fit the Vasyliunas-Siscoe helium PUI model to one 10-minute chunk.

    Parameters
    ----------
    esa_energies : ndarray of floats
        The (50, 72) array of L2 `esa_energy` values. Only the 62 coarse steps
        are used; the fine sweep and the discarded bin 0 are dropped here.
    count_rates : ndarray of floats
        The (50, 72) array of coincidence count rates, on the same full sweep.
    bulk_sw_per_bin_swapi_kms : ndarray of floats
        The (50, 62, 3) array of bulk solar wind velocity vectors in the SWAPI
        frame, at each coarse-step measurement time.
    vasyliunas_siscoe_distribution: TODO
    time_as_tt2000: TODO
    """
    if esa_energies.shape != (_SWEEPS_PER_CHUNK, _SWEEP_LEN):
        raise ValueError(esa_energies.shape)

    if count_rates.shape != (_SWEEPS_PER_CHUNK, _SWEEP_LEN):
        raise ValueError(count_rates.shape)

    if bulk_sw_per_bin_swapi_kms.shape != (_SWEEPS_PER_CHUNK, _COARSE_SWEEP_LEN, 3):
        raise ValueError(bulk_sw_per_bin_swapi_kms.shape)

    coarse_energies = np.abs(
        np.asarray(esa_energies[:, SWAPI_COARSE_SWEEP_BINS], dtype=float).mean(axis=0)
    )
    coarse_count_rates = np.asarray(
        count_rates[:, SWAPI_COARSE_SWEEP_BINS], dtype=float
    )

    lower_energy_cutoff, upper_energy_cutoff = _calculate_pickup_ion_fit_energy_range(
        coarse_energies, coarse_count_rates.mean(axis=0)
    )

    bin_mask = (coarse_energies > lower_energy_cutoff) & (
        coarse_energies < upper_energy_cutoff
    )

    sw_velocity_kms = float(np.linalg.norm(bulk_sw_per_bin_swapi_kms, axis=-1).mean())

    # precomputed instrument response across all coarse ESA steps
    coarse_voltages = coarse_energies / SWAPI_L2_K_FACTOR
    full_sweep_response = build_chunk_collapsed_response(
        swapi_response=swapi_response,
        voltages_v=coarse_voltages,
        bulk_sw_per_bin_kms=bulk_sw_per_bin_swapi_kms,
        time_as_tt2000=time_as_tt2000,
        species=_PICKUP_ION_SPECIES,
        cutoff_speed_max_kms=sw_velocity_kms * 1.2,
    )

    # subset used for fitting
    fit_window_response = ChunkCollapsedResponse(
        speed_in_sw_frame=full_sweep_response.speed_in_sw_frame,
        bin_weights=full_sweep_response.bin_weights[:, bin_mask],
    )

    fitting_params = _fit_pickup_ion_parameters(
        chunk_response=fit_window_response,
        vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
        observed_count_rates=coarse_count_rates[:, bin_mask],
        sw_speed_kms=sw_velocity_kms,
    )

    if not (int(fitting_params.flags) & int(SwapiL3Flags.BAD_FIT)):
        nominal_params = FittingParameters(*map(nominal_value, astuple(fitting_params)))
        full_sweep_model_rates = calculate_coincidence_rate(
            full_sweep_response, vasyliunas_siscoe_distribution, nominal_params
        )
        if not is_good_fit(
            esa_energies=coarse_energies,
            model_rates=full_sweep_model_rates,
            observed_rates=coarse_count_rates,
            cutoff_speed_kms=nominal_params.cutoff_speed,
            min_fitting_energy=lower_energy_cutoff,
        ):
            nan_param = ufloat(np.nan, np.nan)
            fitting_params = FittingParameters(
                nan_param,
                nan_param,
                nan_param,
                fitting_params.flags | SwapiL3Flags.BAD_FIT,
            )

    return PickupIonFitResult(
        fitting_params=fitting_params,
        chunk_response=fit_window_response,
        vasyliunas_siscoe_distribution=vasyliunas_siscoe_distribution,
    )


def _calculate_pickup_ion_fit_energy_range(
    energies_per_step: ndarray, count_rates_per_step: ndarray
) -> tuple[float, float]:
    proton_peak_energy = energies_per_step[np.argmax(count_rates_per_step)]

    # assumes alpha solar wind has the same bulk speed as proton solar wind
    nominal_alpha_peak = 2 * proton_peak_energy

    # assumes that the PUI cutoff speed is 2x the solar wind speed (4x the energy) in the SC frame
    # accounts for the 4x mass per charge of He+ compared to protons
    # together, that's a factor of 2^2*4=4x4=16
    nominal_pui_he_cutoff = 16 * proton_peak_energy

    # geometric mean (logarithmic midpoint) between estimated alpha peak and nominal PUI peak
    lower_edge = float(np.sqrt(nominal_alpha_peak * nominal_pui_he_cutoff))

    # use nominal PUI cutoff as the upper edge for the fitting range
    upper_edge = float(nominal_pui_he_cutoff)

    return float(lower_edge), float(upper_edge)


def _fit_pickup_ion_parameters(
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
    observed_count_rates: np.ndarray,
    sw_speed_kms: float,
) -> FittingParameters:
    """Run the Nelder-Mead PUI parameter fit.

    `observed_count_rates` is shape (n_sweeps, n_steps). `chunk_response` and
    `vasyliunas_siscoe_distribution` carry the precomputed geometry; the
    residual constructs a `FittingParameters` from each iteration's lmfit values.
    """
    params = Parameters()
    params.add("cooling_index", value=1.5, min=1.0, max=5.0)
    params.add("ionization_rate", value=1e-7, min=0.6e-9, max=8.0e-7)
    params.add(
        "cutoff_speed",
        value=sw_speed_kms,
        min=sw_speed_kms * 0.8,
        max=sw_speed_kms * 1.2,
    )

    def map_to_internal(value, param):
        return np.arcsin(2 * (value - param.min) / (param.max - param.min) - 1)

    def simplex_vertex(cooling_index, ionization_rate, cutoff_speed):
        return [
            map_to_internal(cooling_index, params["cooling_index"]),
            map_to_internal(ionization_rate, params["ionization_rate"]),
            map_to_internal(cutoff_speed, params["cutoff_speed"]),
        ]

    initial_simplex = np.array(
        [
            simplex_vertex(1.5, 1e-7, sw_speed_kms),
            simplex_vertex(5.0, 1e-7, sw_speed_kms),
            simplex_vertex(1.5, 2.1e-7, sw_speed_kms),
            simplex_vertex(1.5, 1e-7, sw_speed_kms * 1.2),
        ]
    )

    minimizer = lmfit.Minimizer(
        _calculate_poisson_negative_log_likelihood,
        params,
        fcn_args=(observed_count_rates, chunk_response, vasyliunas_siscoe_distribution),
        scale_covar=False,
        options=dict(initial_simplex=initial_simplex),
    )
    result = minimizer.minimize(method="nelder")

    nominal_values = result.params.valuesdict()

    flags = SwapiL3Flags.NONE
    hessian_fn = ndt.Hessian(minimizer.penalty)
    try:
        hessian_value = hessian_fn(result.x)
        cov_internal = inv(hessian_value)
        cov_external = minimizer._int2ext_cov_x(cov_internal, result.x)
        standard_errors = np.sqrt(np.diag(cov_external))  # NaN if not positive definite
    except Exception:
        standard_errors = np.full(len(result.var_names), np.nan)

    if not np.all(np.isfinite(standard_errors)):
        flags |= SwapiL3Flags.BAD_FIT

    if flags & SwapiL3Flags.BAD_FIT:
        nan_param = ufloat(np.nan, np.nan)
        return FittingParameters(nan_param, nan_param, nan_param, flags)

    param_vals = {
        name: ufloat(nominal_values[name], std_err)
        for name, std_err in zip(result.var_names, standard_errors)
    }

    return FittingParameters(
        param_vals["cooling_index"],
        param_vals["ionization_rate"],
        param_vals["cutoff_speed"],
        flags,
    )


def _calculate_poisson_negative_log_likelihood(
    params: Parameters,
    observed_count_rates: np.ndarray,  # (n_sweeps, n_steps)
    chunk_response: ChunkCollapsedResponse,
    vasyliunas_siscoe_distribution: VasyliunasSiscoeDistribution,
) -> float:
    parvals = params.valuesdict()
    fitting_params = FittingParameters(
        cooling_index=parvals["cooling_index"],
        ionization_rate=parvals["ionization_rate"],
        cutoff_speed=parvals["cutoff_speed"],
    )

    modeled_rates = (
        calculate_coincidence_rate(
            chunk_response, vasyliunas_siscoe_distribution, fitting_params
        )
        + SWAPI_BACKGROUND_RATE
    )
    modeled_counts = modeled_rates * swapi_l2.SWAPI_LIVETIME
    observed_counts = observed_count_rates * swapi_l2.SWAPI_LIVETIME
    return float(np.sum(modeled_counts - observed_counts * np.log(modeled_counts)))

