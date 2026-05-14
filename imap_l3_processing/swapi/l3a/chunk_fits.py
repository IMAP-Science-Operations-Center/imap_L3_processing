import logging
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from spacepy import pycdf
from uncertainties import ufloat

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    AlphaSolarWindFitResult,
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    ProtonSolarWindFitResult,
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.constants import (
    SWAPI_COARSE_SWEEP_BINS,
    SWAPI_L2_K_FACTOR,
    SWAPI_SCIENCE_BINS,
)
from imap_l3_processing.swapi.l3a.utils import (
    chunk_epoch,
    compute_direction_of_mean_magnetic_field_over_chunk,
    esa_voltage_to_proton_speed,
    get_spacecraft_velocity_rtn,
    get_swapi_geometry,
    measurement_times,
)
from imap_l3_processing.swapi.quality_flags import SwapiL3Flags

logger = logging.getLogger(__name__)

_shared: dict[str, Any] = {}


class ChunkFitter(ABC):
    @abstractmethod
    def precompute_geometry(self, chunk) -> tuple: ...

    @abstractmethod
    def fit_chunk(self, chunk, *geometry) -> dict[str, Any]: ...


class ParallelChunkRunner:
    def __init__(self, swapi_response, efficiency_table):
        self._swapi_response = swapi_response
        self._efficiency_table = efficiency_table

    def run(self, chunks, fitter: ChunkFitter) -> dict[str, np.ndarray]:
        geometries = [fitter.precompute_geometry(chunk) for chunk in chunks]

        with multiprocessing.get_context("fork").Pool(
            processes=os.cpu_count(),
            initializer=_init_worker,
            initargs=(self._swapi_response, self._efficiency_table, fitter),
        ) as pool:
            results = pool.starmap(
                _run_one, [(chunk, geom) for chunk, geom in zip(chunks, geometries)]
            )

        return {k: np.array([r[k] for r in results]) for k in results[0].keys()}


def _init_worker(swapi_response, efficiency_table, fitter):
    _shared["swapi_response"] = swapi_response
    _shared["efficiency_table"] = efficiency_table
    _shared["fitter"] = fitter


def _run_one(chunk, geom):
    return _shared["fitter"].fit_chunk(chunk, *geom)


class ProtonChunkFitter(ChunkFitter):
    def precompute_geometry(self, chunk):
        epoch = chunk_epoch(chunk)
        rm = None
        sc_vel = None
        try:
            rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
        except Exception:
            logger.warning(
                "SPICE gap in rotation matrices.",
                exc_info=True,
            )

        if rm is not None:
            try:
                sc_vel = get_spacecraft_velocity_rtn(epoch)
            except Exception:
                logger.warning(
                    "SPICE gap in spacecraft velocity.",
                    exc_info=True,
                )

        return (epoch, rm, sc_vel)

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        rotation_matrices,
        sc_velocity_rtn,
    ):
        result = _fit_proton(data_chunk, epoch, rotation_matrices)
        return _proton_moments_from_fit(result, epoch, data_chunk, sc_velocity_rtn)


class AlphaChunkFitter(ChunkFitter):
    def __init__(self, mag_data):
        self.mag_data = mag_data

    def precompute_geometry(self, chunk):
        epoch = chunk_epoch(chunk)
        try:
            rm = get_swapi_geometry(measurement_times(chunk, SWAPI_SCIENCE_BINS))
        except Exception:
            logger.info(
                f"Missing SPICE information at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}, continuing with fill value"
            )
            rm = None
        b_hat = compute_direction_of_mean_magnetic_field_over_chunk(
            self.mag_data, int(epoch), int(THIRTY_SECONDS_IN_NANOSECONDS)
        )
        return (epoch, rm, b_hat)

    def fit_chunk(
        self,
        data_chunk,
        epoch,
        rotation_matrices,
        magnetic_field_direction,
    ):
        result = _fit_alpha(
            data_chunk, epoch, rotation_matrices, magnetic_field_direction
        )
        return _alpha_moments_from_fit(result, epoch)


def _proton_moments_from_fit(result, epoch, data_chunk, sc_velocity_rtn):
    speed = sum(component**2 for component in result.bulk_velocity_rtn) ** 0.5
    speed_nom, speed_unc = speed.nominal_value, speed.std_dev
    bulk_velocity_rtn_sc = result.bulk_velocity_rtn_nominal()
    velocity_covariance_sc = result.bulk_velocity_rtn_covariance()
    density_nom, density_unc = result.density.nominal_value, result.density.std_dev
    temp_nom, temp_unc = result.temperature.nominal_value, result.temperature.std_dev

    if sc_velocity_rtn is not None:
        bulk_velocity_rtn_sun = bulk_velocity_rtn_sc + sc_velocity_rtn
        velocity_covariance_sun = velocity_covariance_sc
        sun_velocity_unc = [
            component + sc_component
            for component, sc_component in zip(
                result.bulk_velocity_rtn, sc_velocity_rtn
            )
        ]
        sun_speed = sum(component**2 for component in sun_velocity_unc) ** 0.5
        sun_speed_nom, sun_speed_unc = sun_speed.nominal_value, sun_speed.std_dev
    else:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing spacecraft velocity; sun-frame outputs are fill values"
        )
        bulk_velocity_rtn_sun = np.full(3, np.nan)
        velocity_covariance_sun = np.full((3, 3), np.nan)
        sun_speed_nom = sun_speed_unc = np.nan

    if not np.isfinite(speed_nom):
        speed_nom = _peak_proton_speed_kms(data_chunk)

    return dict(
        epoch=epoch,
        proton_sw_speed=speed_nom,
        proton_sw_speed_uncert=speed_unc,
        proton_sw_speed_sun=sun_speed_nom,
        proton_sw_speed_sun_uncert=sun_speed_unc,
        proton_sw_temperature=temp_nom,
        proton_sw_temperature_uncert=temp_unc,
        proton_sw_density=density_nom,
        proton_sw_density_uncert=density_unc,
        proton_sw_bulk_velocity_rtn_sun=bulk_velocity_rtn_sun,
        proton_sw_bulk_velocity_rtn_sun_covariance=velocity_covariance_sun,
        proton_sw_bulk_velocity_rtn_sc=bulk_velocity_rtn_sc,
        proton_sw_bulk_velocity_rtn_sc_covariance=velocity_covariance_sc,
        quality_flags=result.bad_fit_flag,
    )


def _alpha_moments_from_fit(result, epoch):
    alpha = result.alpha_moments
    speed = sum(component**2 for component in alpha.bulk_velocity_rtn) ** 0.5
    return dict(
        epoch=epoch,
        alpha_sw_speed=speed.nominal_value,
        alpha_sw_speed_uncert=speed.std_dev,
        alpha_sw_density=alpha.density.nominal_value,
        alpha_sw_density_uncert=alpha.density.std_dev,
        alpha_sw_temperature=alpha.temperature.nominal_value,
        alpha_sw_temperature_uncert=alpha.temperature.std_dev,
        alpha_sw_velocity_rtn=alpha.bulk_velocity_rtn_nominal(),
        alpha_sw_velocity_covariance_rtn=alpha.bulk_velocity_rtn_covariance(),
        alpha_sw_delta_v=alpha.delta_v.nominal_value,
        alpha_sw_delta_v_uncert=alpha.delta_v.std_dev,
        alpha_sw_b_hat_rtn=result.b_hat_rtn,
        quality_flags=result.bad_fit_flag,
    )


_PEAK_SPEED_HALF_WIDTH = 4


def _peak_proton_speed_kms(data_chunk) -> float:
    count_rate = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
    voltage = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
    mean_count_rate = np.nanmean(count_rate, axis=0)
    mean_voltage = np.nanmean(voltage, axis=0)
    if not np.any(np.isfinite(mean_count_rate)):
        return np.nan
    peak_index = int(np.nanargmax(mean_count_rate))
    n_bins = mean_count_rate.shape[0]
    clamped_peak_index = max(
        _PEAK_SPEED_HALF_WIDTH, min(peak_index, n_bins - _PEAK_SPEED_HALF_WIDTH - 1)
    )
    window = slice(
        clamped_peak_index - _PEAK_SPEED_HALF_WIDTH,
        clamped_peak_index + _PEAK_SPEED_HALF_WIDTH + 1,
    )
    rate_window = mean_count_rate[window]
    voltage_window = mean_voltage[window]
    valid = (
        np.isfinite(rate_window)
        & np.isfinite(voltage_window)
        & (voltage_window > 0.0)
        & (rate_window >= 0.0)
    )
    if not np.any(valid):
        return np.nan
    speed = esa_voltage_to_proton_speed(voltage_window[valid])
    weights = rate_window[valid]
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.nan
    return float(np.sum(weights * speed) / weight_sum)


def _eff_scale(efficiency_table, epoch, kind):
    eps_lab = float(efficiency_table.eps_p_lab)
    if kind == "proton":
        return float(efficiency_table.get_proton_efficiency_for(epoch)) / eps_lab
    return float(efficiency_table.get_alpha_efficiency_for(epoch)) / eps_lab


def _coarse_subset_of_science_rotations(
    science_rotation_matrices: np.ndarray, n_sweeps: int
) -> np.ndarray:
    """Slice per-sweep coarse-bin rotations out of a flat science-bin rotation
    array. Coarse bins are the first 62 of the 71 science bins per sweep."""
    n_science_bins = SWAPI_SCIENCE_BINS.stop - SWAPI_SCIENCE_BINS.start
    n_coarse_bins = SWAPI_COARSE_SWEEP_BINS.stop - SWAPI_COARSE_SWEEP_BINS.start
    return (
        science_rotation_matrices.reshape(n_sweeps, n_science_bins, 3, 3)[
            :, :n_coarse_bins
        ]
        .reshape(-1, 3, 3)
    )


def _fit_proton(
    data_chunk, epoch, rotation_matrices
) -> ProtonSolarWindFitResult:
    if rotation_matrices is None:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing rotation matrices; using fill values"
        )
        return _nan_proton_result(SwapiL3Flags.NONE)
    if np.any(np.isnan(data_chunk.coincidence_count_rate[:, SWAPI_SCIENCE_BINS])):
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: NaN in input count rate; using fill values"
        )
        return _nan_proton_result(SwapiL3Flags.NONE)
    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    count_rates = data_chunk.coincidence_count_rate[:, SWAPI_SCIENCE_BINS]
    voltages = data_chunk.energy[:, SWAPI_SCIENCE_BINS] / SWAPI_L2_K_FACTOR
    ctx = build_solar_wind_fit_context(
        count_rate=count_rates,
        esa_voltage=voltages,
        swapi_response=swapi_response,
        central_effective_area_scale=_eff_scale(efficiency_table, epoch, "proton"),
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    try:
        result = fit_solar_wind_proton_model(ctx)
    except Exception:
        logger.warning(
            f"Proton fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
            exc_info=True,
        )
        return _nan_proton_result(SwapiL3Flags.FIT_ERROR)

    return result


def _nan_proton_result(flag) -> ProtonSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return ProtonSolarWindFitResult(
        density=nan,
        temperature=nan,
        bulk_velocity_rtn=(nan, nan, nan),
        bad_fit_flag=int(flag),
    )


@dataclass
class AlphaChunkFitResult:
    """Chunk-level alpha output: alpha moments + the proton moments they
    reference + the B̂ used as the field-aligned drift constraint, plus the
    consolidated `bad_fit_flag` rolled up from every stage that could fail
    (missing B̂, proton fit, alpha LM)."""

    alpha_moments: AlphaSolarWindFitResult
    proton_moments: ProtonSolarWindFitResult
    b_hat_rtn: np.ndarray
    bad_fit_flag: int


def _nan_alpha_fit_result(flag) -> AlphaSolarWindFitResult:
    nan = ufloat(np.nan, np.nan)
    return AlphaSolarWindFitResult(
        density=nan,
        temperature=nan,
        bulk_velocity_rtn=(nan, nan, nan),
        delta_v=nan,
        bad_fit_flag=int(flag),
    )


def _fit_alpha(
    data_chunk,
    epoch,
    rotation_matrices,
    magnetic_field_direction,
) -> AlphaChunkFitResult:
    nan_b_hat = np.full(3, np.nan)
    if (
        magnetic_field_direction is None
        or not np.all(np.isfinite(magnetic_field_direction))
    ):
        logger.warning(
            f"Alpha fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: missing or non-finite magnetic field direction; using fill values"
        )
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(SwapiL3Flags.NONE),
            proton_moments=_nan_proton_result(SwapiL3Flags.NONE),
            b_hat_rtn=nan_b_hat,
            bad_fit_flag=int(SwapiL3Flags.NONE),
        )

    proton_moments = _fit_proton(data_chunk, epoch, rotation_matrices)
    if not np.all(
        np.isfinite(
            [component.nominal_value for component in proton_moments.bulk_velocity_rtn]
        )
    ):
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(proton_moments.bad_fit_flag),
            proton_moments=proton_moments,
            b_hat_rtn=nan_b_hat,
            bad_fit_flag=int(proton_moments.bad_fit_flag),
        )

    swapi_response = _shared["swapi_response"]
    efficiency_table = _shared["efficiency_table"]
    try:
        count_rates = data_chunk.coincidence_count_rate[:, SWAPI_COARSE_SWEEP_BINS]
        voltages = data_chunk.energy[:, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
        coarse_rotation_matrices = _coarse_subset_of_science_rotations(
            rotation_matrices, n_sweeps=data_chunk.sci_start_time.shape[0]
        )
        proton_ctx = build_solar_wind_fit_context(
            count_rate=count_rates,
            esa_voltage=voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=_eff_scale(efficiency_table, epoch, "proton"),
            rotation_matrices=coarse_rotation_matrices,
            mass_kg=PROTON_MASS_KG,
            mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_ctx = build_solar_wind_fit_context(
            count_rate=count_rates,
            esa_voltage=voltages,
            swapi_response=swapi_response,
            central_effective_area_scale=_eff_scale(efficiency_table, epoch, "alpha"),
            rotation_matrices=coarse_rotation_matrices,
            mass_kg=ALPHA_PARTICLE_MASS_KG,
            mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
        )
        alpha_moments = fit_solar_wind_alpha_model(
            proton_ctx=proton_ctx,
            alpha_ctx=alpha_ctx,
            proton_moments=proton_moments,
            magnetic_field_direction=magnetic_field_direction,
        )
    except Exception:
        logger.warning(
            f"Alpha fit at epoch {pycdf.lib.tt2000_to_datetime(int(epoch))}: exception during fit; using fill values",
            exc_info=True,
        )
        return AlphaChunkFitResult(
            alpha_moments=_nan_alpha_fit_result(SwapiL3Flags.FIT_ERROR),
            proton_moments=proton_moments,
            b_hat_rtn=nan_b_hat,
            bad_fit_flag=int(SwapiL3Flags.FIT_ERROR),
        )

    return AlphaChunkFitResult(
        alpha_moments=alpha_moments,
        proton_moments=proton_moments,
        b_hat_rtn=magnetic_field_direction,
        bad_fit_flag=int(alpha_moments.bad_fit_flag),
    )
