import numpy as np
from numpy import ndarray
from scipy.stats import circstd
import uncertainties.umath as umath
import uncertainties.unumpy as unp
from uncertainties import UFloat, correlated_values, covariance_matrix, ufloat

from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    SolarWindFitContext,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.state import (
    LOG_DENSITY_IDX,
    LOG_TEMPERATURE_IDX,
    VELOCITY_SLICE,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.optimizer import (
    OptimizeSolarWindParamsResult,
)


N_VELOCITY_ANGLE_MC_SAMPLES = 100


def derive_uncertainties(
    result: OptimizeSolarWindParamsResult,
    ctx: SolarWindFitContext,
) -> tuple[float, float, ndarray]:
    residual_jacobian = result.jacobian
    try:
        JT_J_pseudoinverse = np.linalg.pinv(residual_jacobian.T @ residual_jacobian)
        # HC3 inner factor: r_i^2 / (1 - h_ii)^2 where h_ii are hat-matrix
        # diagonals. The (1 - h_ii)^-2 reweight corrects for the finite-sample
        # bias that LM constraints (J^T r = 0) induce on r_i^2 at high-leverage
        # bins — a few bins at the proton peak carry most of the parameter
        # information, so their leverage is large and HC0 understates by ~20%.
        # Hayes & Cai (2007) recommends HC3 for small effective N.
        leverage = np.einsum(
            "ki,ij,kj->k",
            residual_jacobian,
            JT_J_pseudoinverse,
            residual_jacobian,
        )
        leverage_clipped = np.clip(leverage, 0.0, 0.9999)
        hc3_weights = (result.residuals / (1.0 - leverage_clipped)) ** 2
        sandwich_middle = np.einsum(
            "ki,k,kj->ij", residual_jacobian, hc3_weights, residual_jacobian
        )
        parameter_covariance = JT_J_pseudoinverse @ sandwich_middle @ JT_J_pseudoinverse

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
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.full((3, 3), np.nan)

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


def derive_velocity_angles(
    bulk_velocity_rtn: tuple[UFloat, UFloat, UFloat],
    epoch_tt2000_ns: float,
) -> tuple:
    from imap_l3_processing.swapi.l3a.utils import rotate_rtn_to_dps

    velocity_dps_unc = rotate_rtn_to_dps(np.array(bulk_velocity_rtn), epoch_tt2000_ns)
    velocity_dps = unp.nominal_values(velocity_dps_unc)
    velocity_dps_cov = np.array(covariance_matrix(velocity_dps_unc))

    speed_nominal = float(np.linalg.norm(velocity_dps))
    clock_nominal = float(
        np.degrees(np.arctan2(-velocity_dps[1], -velocity_dps[0])) % 360
    )
    deflection_nominal = float(np.degrees(np.arccos(-velocity_dps[2] / speed_nominal)))

    if not np.all(np.isfinite(velocity_dps_cov)):
        return (
            ufloat(speed_nominal, np.nan),
            ufloat(clock_nominal, np.nan),
            ufloat(deflection_nominal, np.nan),
        )

    speed = umath.sqrt(sum(x**2 for x in velocity_dps_unc))
    clock_sigma, deflection_sigma = _clock_and_deflection_sigmas_via_monte_carlo(
        velocity_dps, velocity_dps_cov
    )

    return (
        speed,
        ufloat(clock_nominal, clock_sigma),
        ufloat(deflection_nominal, deflection_sigma),
    )


def _clock_and_deflection_sigmas_via_monte_carlo(
    velocity_mean: ndarray, velocity_cov: ndarray
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(
        velocity_mean,
        velocity_cov,
        size=N_VELOCITY_ANGLE_MC_SAMPLES,
        check_valid="ignore",
    )

    sample_clocks = np.degrees(np.arctan2(-samples[:, 1], -samples[:, 0])) % 360
    clock_sigma = float(circstd(sample_clocks, high=360))

    sample_speeds = np.linalg.norm(samples, axis=1)
    sample_deflections = np.degrees(
        np.arccos(np.clip(-samples[:, 2] / sample_speeds, -1, 1))
    )
    deflection_sigma = float(np.std(sample_deflections, ddof=1))

    return clock_sigma, deflection_sigma
