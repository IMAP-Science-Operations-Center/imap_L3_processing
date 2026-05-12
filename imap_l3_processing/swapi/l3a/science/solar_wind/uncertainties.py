import numpy as np
from numpy import ndarray
from uncertainties import UFloat, correlated_values, ufloat

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
)


def compute_hc3_parameter_covariance(
    jacobian: ndarray, residuals: ndarray
) -> ndarray:
    n_params = jacobian.shape[1]
    try:
        JT_J_pseudoinverse = np.linalg.pinv(jacobian.T @ jacobian)
        h_ii = np.einsum(
            "ki,ij,jk->k",
            jacobian,
            JT_J_pseudoinverse,
            jacobian.T,
        )
        h_ii_clipped = np.clip(h_ii, 0.0, 0.9999)
        hc3_weights = (residuals / (1.0 - h_ii_clipped)) ** 2
        sandwich_middle = np.einsum(
            "ki,k,kj->ij", jacobian, hc3_weights, jacobian
        )
        return JT_J_pseudoinverse @ sandwich_middle @ JT_J_pseudoinverse
    except np.linalg.LinAlgError:
        return np.full((n_params, n_params), np.nan)


def r_squared(residuals: ndarray, data: ndarray) -> float:
    data_mean = float(np.mean(data))
    ss_tot = float(np.sum((data - data_mean) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum(residuals**2))
    return 1.0 - ss_res / ss_tot


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


def make_correlated_velocity(
    nominal: ndarray, covariance: ndarray
) -> tuple[UFloat, UFloat, UFloat]:
    is_finite = np.all(np.isfinite(covariance))
    is_positive_semidefinite = is_finite and np.linalg.eigvalsh(covariance)[0] >= 0
    if not is_positive_semidefinite:
        return tuple(ufloat(float(v), np.nan) for v in nominal)

    return tuple(correlated_values(nominal, covariance))
