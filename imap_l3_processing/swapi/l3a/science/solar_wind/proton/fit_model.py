from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    SolarWindParams,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.basin_hopping import (
    flipped_seed,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.initial_guess import (
    calculate_initial_guess,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
    r_squared,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.utils import average_spin_axis_rtn
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags
from imap_l3_processing.swapi.response.deadtime import deadtime_factor


_R_SQUARED_PEAK_HALF_WIDTH = 4
_MAX_BASIN_REFINE_ITERS = 6
_ROTATED_RMSE_RATIO_THRESHOLD = 10


@dataclass
class OptimizeSolarWindParamsResult:
    sw_params: SolarWindParams
    residuals: ndarray  # count-rate residuals at the solution
    jacobian: ndarray  # ∂residuals/∂state, columns ordered per the state vector
    success: bool

    @property
    def mse(self) -> float:
        return float(np.mean(self.residuals**2))


def optimize_solar_wind_params(
    initial_guess: SolarWindParams, ctx: SolarWindFitContext
) -> OptimizeSolarWindParamsResult:
    evaluator = _Evaluator(ctx)

    raw: scipy.optimize.OptimizeResult = scipy.optimize.least_squares(
        evaluator.residues,
        initial_guess.to_vector(),
        jac=evaluator.jacobian,
        method="lm",
        xtol=1e-4,
        x_scale=1.0,
    )

    return OptimizeSolarWindParamsResult(
        sw_params=SolarWindParams.from_vector(raw.x, ctx.mass_kg),
        residuals=raw.fun,
        jacobian=raw.jac,
        success=bool(raw.success),
    )


class _Evaluator:
    """Caches the most recent (residuals, jacobian) so scipy.least_squares' separate
    `fun` and `jac` callbacks share a single forward-model evaluation per state."""

    def __init__(self, ctx: SolarWindFitContext):
        self.ctx = ctx
        self._last_state: ndarray | None = None
        self._last_residues: ndarray | None = None
        self._last_jacobian: ndarray | None = None

    def _eval(self, state: ndarray) -> None:
        sw = SolarWindParams.from_vector(state, self.ctx.mass_kg)
        rate_ideal, jacobian_ideal = model_solar_wind_ideal_coincidence_rates(
            sw, self.ctx
        )

        df = deadtime_factor(rate_ideal)

        rate_observable = rate_ideal * df
        jacobian = jacobian_ideal * np.square(df)[:, np.newaxis]

        residues = rate_observable - self.ctx.count_rate.ravel()

        self._last_state = state.copy()
        self._last_residues = residues
        self._last_jacobian = jacobian

    def _refresh(self, state: ndarray) -> None:
        if self._last_state is None or not np.array_equal(state, self._last_state):
            self._eval(state)

    def residues(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_residues

    def jacobian(self, state: ndarray) -> ndarray:
        self._refresh(state)
        return self._last_jacobian


@dataclass
class ProtonSolarWindFitResult:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    bad_fit_flag: int

    def bulk_velocity_rtn_nominal(self) -> ndarray:
        return np.array([v.nominal_value for v in self.bulk_velocity_rtn])

    def bulk_velocity_rtn_covariance(self) -> ndarray:
        return np.array(covariance_matrix(self.bulk_velocity_rtn))


def fit_solar_wind_proton_model(ctx: SolarWindFitContext) -> ProtonSolarWindFitResult:
    initial_guess = calculate_initial_guess(ctx)
    first_result = optimize_solar_wind_params(initial_guess, ctx)
    final_result = escape_local_minimum(first_result, ctx)
    return _construct_fit_result(final_result, ctx)


def escape_local_minimum(
    first_result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> OptimizeSolarWindParamsResult:
    spin_axis_rtn = average_spin_axis_rtn(ctx.rotation_matrices)

    current_result = first_result
    for _ in range(_MAX_BASIN_REFINE_ITERS):
        flipped_mse, flipped_seed_params = flipped_seed(
            current_result, ctx, spin_axis_rtn
        )

        flipped_seed_is_substantially_worse_than_current = (
            flipped_mse >= current_result.mse * _ROTATED_RMSE_RATIO_THRESHOLD**2
        )
        if flipped_seed_is_substantially_worse_than_current:
            break

        flipped_result = optimize_solar_wind_params(flipped_seed_params, ctx)

        flipped_result_is_worse_than_current = flipped_result.mse > current_result.mse
        if flipped_result_is_worse_than_current:
            break

        current_result = flipped_result

    return current_result


def _construct_fit_result(final_result, ctx):
    if not final_result.success:
        return _nan_proton_fit_result(int(SwapiL3Flags.FIT_ERROR))

    fit_r_squared = _coarse_peak_averaged_r_squared(
        final_result.residuals, ctx.count_rate
    )
    flag_bad = (
        fit_r_squared < 0.9
        or final_result.sw_params.temperature > 5.0e5
    )
    if flag_bad:
        return _nan_proton_fit_result(int(SwapiL3Flags.BAD_FIT))

    density_sigma, temperature_sigma, velocity_covariance = derive_uncertainties(
        final_result, ctx
    )
    density = ufloat(final_result.sw_params.density, density_sigma)
    temperature = ufloat(final_result.sw_params.temperature, temperature_sigma)
    bulk_velocity_rtn = make_correlated_velocity(
        final_result.sw_params.bulk_velocity_rtn, velocity_covariance
    )
    return ProtonSolarWindFitResult(
        density=density,
        temperature=temperature,
        bulk_velocity_rtn=bulk_velocity_rtn,
        bad_fit_flag=int(SwapiL3Flags.NONE),
    )


def _nan_proton_fit_result(bad_fit_flag: int) -> ProtonSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return ProtonSolarWindFitResult(
        density=nan,
        temperature=nan,
        bulk_velocity_rtn=(nan, nan, nan),
        bad_fit_flag=bad_fit_flag,
    )


def _coarse_peak_averaged_r_squared(residuals: ndarray, count_rate: ndarray) -> float:
    if count_rate.ndim != 2:
        return r_squared(residuals, count_rate)
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    coarse_count_rate_per_sweep = count_rate[:, :n_coarse_bins]
    coarse_residual_per_sweep = residuals.reshape(count_rate.shape)[:, :n_coarse_bins]
    averaged_count_rate = np.nanmean(coarse_count_rate_per_sweep, axis=0)
    averaged_residual = np.nanmean(coarse_residual_per_sweep, axis=0)
    peak_index = int(np.nanargmax(averaged_count_rate))
    n_bins = averaged_count_rate.shape[0]
    clamped_peak_index = max(
        _R_SQUARED_PEAK_HALF_WIDTH,
        min(peak_index, n_bins - _R_SQUARED_PEAK_HALF_WIDTH - 1),
    )
    window = slice(
        clamped_peak_index - _R_SQUARED_PEAK_HALF_WIDTH,
        clamped_peak_index + _R_SQUARED_PEAK_HALF_WIDTH + 1,
    )
    return r_squared(averaged_residual[window], averaged_count_rate[window])


def derive_uncertainties(
    result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> tuple[float, float, ndarray]:
    parameter_covariance = compute_hc3_parameter_covariance(
        result.jacobian, result.residuals
    )
    if not np.all(np.isfinite(parameter_covariance)):
        return np.nan, np.nan, np.full((3, 3), np.nan)

    log_density_variance = parameter_covariance[LOG_DENSITY_IDX, LOG_DENSITY_IDX]
    log_temperature_variance = parameter_covariance[
        LOG_TEMPERATURE_IDX, LOG_TEMPERATURE_IDX
    ]

    density_error = float(
        result.sw_params.density * np.sqrt(max(log_density_variance, 0.0))
    )
    temperature_error = float(
        result.sw_params.temperature * np.sqrt(max(log_temperature_variance, 0.0))
    )
    velocity_covariance = parameter_covariance[VELOCITY_SLICE, VELOCITY_SLICE]

    return (
        density_error,
        temperature_error,
        velocity_covariance,
    )
