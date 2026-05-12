from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.optimize
from numpy import ndarray
from uncertainties import UFloat, covariance_matrix, ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_model import (
    ProtonSolarWindFitResult,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    SolarWindParams,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.uncertainties import (
    compute_hc3_parameter_covariance,
    make_correlated_velocity,
    r_squared,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.utils import (
    get_alpha_peak_indices,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_K_FACTOR,
)
from imap_l3_processing.swapi.response.swapi_response import SwapiResponse
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags


@dataclass
class AlphaSolarWindMoments:
    density: UFloat  # cm^-3
    temperature: UFloat  # K
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat]  # km/s, [R, T, N]; correlated
    delta_v: UFloat  # km/s, signed; +Δv ⇔ alpha drifts along +B̂ vs proton frame
    bad_fit_flag: int

    def bulk_velocity_rtn_nominal(self) -> ndarray:
        """Nominal RTN velocity vector (km/s); shape (3,)."""
        return np.array([v.nominal_value for v in self.bulk_velocity_rtn])

    def bulk_velocity_rtn_covariance(self) -> ndarray:
        """3×3 RTN velocity covariance (km²/s²); = Σ_vp + σ_Δv² B̂B̂ᵀ."""
        return np.array(covariance_matrix(self.bulk_velocity_rtn))


def _nan_alpha_moments(flag: int) -> AlphaSolarWindMoments:
    nan = ufloat(np.nan, np.nan)
    return AlphaSolarWindMoments(
        density=nan,
        temperature=nan,
        bulk_velocity_rtn=(nan, nan, nan),
        delta_v=nan,
        bad_fit_flag=int(flag),
    )


class _AlphaEvaluator:
    """Computes Stage-2 residuals and the analytic 3-column Jacobian in
    (log n_α, log T_α, Δv) space from a single forward-model evaluation,
    caching the last (residuals, jacobian) so scipy's separate `fun` and
    `jac` callbacks share one call per state.

    Builds the 3-D Jacobian from the 5-D forward-model Jacobian
    (∂R_α/∂[log n_α, log T_α, v_R, v_T, v_N]) by the chain rule on
    v_α = v_p + Δv·B̂, then folds the deadtime chain factor 𝒟² for the
    summed proton+alpha observable (see `docs/swapi/solar-wind-moments.md`
    § Deadtime correction)."""

    def __init__(
        self,
        proton_bulk: ndarray,
        magnetic_field_direction: ndarray,
        proton_true_rate: ndarray,
        alpha_ctx: SolarWindFitContext,
    ):
        self.proton_bulk = proton_bulk
        self.magnetic_field_direction = magnetic_field_direction
        self.proton_true_rate = proton_true_rate
        self.alpha_ctx = alpha_ctx
        self._last_state: Optional[ndarray] = None
        self._last_residuals: Optional[ndarray] = None
        self._last_jacobian: Optional[ndarray] = None

    def _eval(self, x: ndarray) -> None:
        n_a = float(np.exp(x[0]))
        T_a = float(np.exp(x[1]))
        delta_v = float(x[2])
        v_a_rtn = self.proton_bulk + delta_v * self.magnetic_field_direction
        alpha_true, jacobian_alpha_5d = model_solar_wind_ideal_coincidence_rates(
            SolarWindParams(n_a, v_a_rtn, T_a, self.alpha_ctx.mass_kg),
            self.alpha_ctx,
        )
        total_true = self.proton_true_rate + alpha_true
        deadtime = deadtime_factor(total_true)
        residuals = total_true * deadtime - self.alpha_ctx.count_rate

        deadtime_squared = deadtime * deadtime
        jacobian = np.empty((alpha_true.size, 3))
        jacobian[:, 0] = deadtime_squared * jacobian_alpha_5d[:, LOG_DENSITY_IDX]
        jacobian[:, 1] = deadtime_squared * jacobian_alpha_5d[:, LOG_TEMPERATURE_IDX]
        jacobian[:, 2] = deadtime_squared * (
            jacobian_alpha_5d[:, VELOCITY_SLICE] @ self.magnetic_field_direction
        )

        self._last_state = x.copy()
        self._last_residuals = residuals
        self._last_jacobian = jacobian

    def _refresh(self, x: ndarray) -> None:
        if self._last_state is None or not np.array_equal(x, self._last_state):
            self._eval(x)

    def residuals(self, x: ndarray) -> ndarray:
        self._refresh(x)
        return self._last_residuals

    def jacobian(self, x: ndarray) -> ndarray:
        self._refresh(x)
        return self._last_jacobian


def fit_solar_wind_alpha_moments(
    count_rate: ndarray,
    esa_voltage: ndarray,
    measurement_time: ndarray,
    swapi_response: SwapiResponse,
    proton_moments: ProtonSolarWindFitResult,
    magnetic_field_direction: ndarray,
    alpha_effective_area_scale: float,
    proton_effective_area_scale: float,
    rotation_matrices: Optional[ndarray] = None,
) -> AlphaSolarWindMoments:
    proton_bad_fit_flag = int(proton_moments.bad_fit_flag)
    proton_bulk_rtn = proton_moments.bulk_velocity_rtn_nominal()

    if not np.all(np.isfinite(proton_bulk_rtn)):
        return _nan_alpha_moments(proton_bad_fit_flag)

    bad_fit_flag = proton_bad_fit_flag

    magnetic_field_direction = np.asarray(magnetic_field_direction, dtype=float)
    if not np.all(np.isfinite(magnetic_field_direction)):
        return _nan_alpha_moments(bad_fit_flag)

    # SPICE shared with Stage 1 if provided; otherwise compute here.
    if rotation_matrices is None:
        from imap_l3_processing.swapi.l3a.utils import get_swapi_geometry

        rotation_matrices = get_swapi_geometry(measurement_time)

    proton_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=swapi_response,
        central_effective_area_scale=proton_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_ctx = build_solar_wind_fit_context(
        count_rate=count_rate,
        esa_voltage=esa_voltage,
        swapi_response=swapi_response,
        central_effective_area_scale=alpha_effective_area_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )

    # Frozen pre-deadtime proton model rate. Deadtime acts on (proton + alpha) below.
    proton_true_rate, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(
            density=proton_moments.density.nominal_value,
            bulk_velocity_rtn=proton_bulk_rtn,
            temperature=proton_moments.temperature.nominal_value,
            mass=proton_ctx.mass_kg,
        ),
        proton_ctx,
    )

    # Initial guess from the alpha bump after subtracting the deadtime-applied proton bg.
    initial_guess = _alpha_initial_guess(
        proton_true_rate=proton_true_rate,
        proton_temperature=proton_moments.temperature.nominal_value,
        alpha_ctx=alpha_ctx,
        proton_bulk_velocity_rtn=proton_bulk_rtn,
        magnetic_field_direction=magnetic_field_direction,
    )
    if initial_guess is None:
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.FIT_ERROR)

    n0, T0, dv0, peak_bin_idx = initial_guess
    proton_bulk = proton_bulk_rtn

    # Subset all per-measurement arrays to only the alpha peak bins across
    # all sweeps, so LM fits the alpha bump rather than the proton-dominated
    # tails (which create an n↓/T↑ degeneracy).
    n_sweeps, n_bins = _infer_sweep_layout(esa_voltage)
    peak_flat_idx = np.concatenate([peak_bin_idx + s * n_bins for s in range(n_sweeps)])
    count_rate_peak = count_rate[peak_flat_idx]
    keep = count_rate_peak > 0
    if not np.all(keep):
        peak_flat_idx = peak_flat_idx[keep]
    proton_true_rate_peak = proton_true_rate[peak_flat_idx]
    alpha_ctx_peak = alpha_ctx.subset(peak_flat_idx)

    evaluator = _AlphaEvaluator(
        proton_bulk=proton_bulk,
        magnetic_field_direction=magnetic_field_direction,
        proton_true_rate=proton_true_rate_peak,
        alpha_ctx=alpha_ctx_peak,
    )

    x0 = np.array([np.log(max(n0, 1e-3)), np.log(max(T0, 1e-3)), dv0])
    result = scipy.optimize.least_squares(
        evaluator.residuals, x0, jac=evaluator.jacobian, method="lm"
    )

    n_a_fit = float(np.exp(result.x[0]))
    T_a_fit = float(np.exp(result.x[1]))
    dv_fit = float(result.x[2])
    bulk_velocity_rtn = proton_bulk + dv_fit * magnetic_field_direction

    if not result.success:
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.FIT_ERROR)

    if r_squared(result.fun, alpha_ctx_peak.count_rate) < 0.9 or T_a_fit > 5.0e5:
        return _nan_alpha_moments(bad_fit_flag | SwapiL3Flags.BAD_FIT)

    cov_x = compute_hc3_parameter_covariance(result.jac, result.fun)
    density_sigma = float(n_a_fit * np.sqrt(max(cov_x[0, 0], 0.0)))
    temperature_sigma = float(T_a_fit * np.sqrt(max(cov_x[1, 1], 0.0)))
    delta_v_sigma = float(np.sqrt(max(cov_x[2, 2], 0.0)))

    sigma_dv2 = max(cov_x[2, 2], 0.0)
    velocity_covariance_rtn = (
        proton_moments.bulk_velocity_rtn_covariance()
        + sigma_dv2 * np.outer(magnetic_field_direction, magnetic_field_direction)
    )

    return AlphaSolarWindMoments(
        density=ufloat(n_a_fit, density_sigma),
        temperature=ufloat(T_a_fit, temperature_sigma),
        bulk_velocity_rtn=make_correlated_velocity(
            bulk_velocity_rtn, velocity_covariance_rtn
        ),
        delta_v=ufloat(dv_fit, delta_v_sigma),
        bad_fit_flag=int(bad_fit_flag),
    )


def _alpha_initial_guess(
    proton_true_rate: ndarray,
    proton_temperature: float,
    alpha_ctx: SolarWindFitContext,
    proton_bulk_velocity_rtn: ndarray,
    magnetic_field_direction: ndarray,
) -> Optional[tuple]:
    """Return (n_α, T_α, Δv=0, peak_bin_indices) as a starting point for LM, or None if peak-finding fails."""

    n_meas = len(alpha_ctx.esa_voltage)
    if n_meas == 0:
        return None

    n_sweeps, n_bins = _infer_sweep_layout(alpha_ctx.esa_voltage)
    if n_sweeps is None:
        return None

    counts_per_sweep = alpha_ctx.count_rate.reshape(n_sweeps, n_bins)
    voltage_per_sweep = alpha_ctx.esa_voltage.reshape(n_sweeps, n_bins)[0]
    proton_true_per_sweep = proton_true_rate.reshape(n_sweeps, n_bins)
    proton_obs_per_sweep = proton_true_per_sweep * deadtime_factor(
        proton_true_per_sweep
    )

    count_avg = counts_per_sweep.mean(axis=0)
    proton_bg_avg = proton_obs_per_sweep.mean(axis=0)
    energies_per_sweep = SWAPI_K_FACTOR * np.abs(voltage_per_sweep)
    proton_peak_index = np.argmax(proton_bg_avg)
    residual = np.maximum(0, count_avg - proton_bg_avg * 2)

    try:
        peak = get_alpha_peak_indices(residual, energies_per_sweep, proton_peak_index)
    except Exception:
        return None

    peak_idx = np.arange(peak.start, peak.stop)
    if len(peak_idx) < 3:
        return None
    residual_peak = np.maximum(residual[peak_idx], 0.0)
    if not np.any(residual_peak > 0):
        return None

    T_alpha = proton_temperature

    unit_alpha, _ = model_solar_wind_ideal_coincidence_rates(
        SolarWindParams(1.0, proton_bulk_velocity_rtn, T_alpha, alpha_ctx.mass_kg),
        alpha_ctx,
    )
    unit_alpha_per_sweep = unit_alpha.reshape(n_sweeps, n_bins).mean(axis=0)
    denom = float(np.nanmean(unit_alpha_per_sweep[peak_idx]))
    if denom <= 0 or not np.isfinite(denom):
        return None
    n_alpha = float(np.nanmean(residual_peak)) / denom
    n_alpha = max(n_alpha, 1e-3)

    return (n_alpha, T_alpha, 0.0, peak_idx)


def _infer_sweep_layout(esa_voltage: ndarray) -> tuple:
    """Heuristic: detect n_sweeps from a periodic voltage axis."""
    n_meas = len(esa_voltage)
    for n_sweeps in (5, 1, 2, 3, 4, 6, 7, 8, 10):
        if n_meas % n_sweeps != 0:
            continue
        n_bins = n_meas // n_sweeps
        first = esa_voltage[:n_bins]
        if np.allclose(esa_voltage.reshape(n_sweeps, n_bins), first):
            return n_sweeps, n_bins
    return None, None
