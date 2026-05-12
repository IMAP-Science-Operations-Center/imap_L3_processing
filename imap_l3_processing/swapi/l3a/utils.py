from datetime import datetime
from typing import Iterable

import numpy as np
import scipy.optimize
import uncertainties
from numpy import ndarray
from spacepy import pycdf
from spacepy.pycdf import CDF
from uncertainties import umath, unumpy

from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.constants import (
    METERS_PER_KILOMETER,
    ONE_SECOND_IN_NANOSECONDS,
    THIRTY_SECONDS_IN_NANOSECONDS,
)
from imap_l3_processing.models import MagData
from imap_l3_processing.swapi.l3a.models import SwapiL2Data
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    get_rotation_matrix,
    imap_state,
)
from imap_processing.spice.time import ttj2000ns_to_et

from imap_l3_processing.swapi.response.deadtime import deadtime_factor


def calculate_sw_speed(particle_mass, particle_charge, energy):
    if np.size(energy) == 0:
        return np.array([])
    dimensions = np.asanyarray(energy).ndim
    if dimensions > 0:
        if isinstance(np.ravel(energy)[0], uncertainties.UFloat):
            return (
                unumpy.sqrt(2 * energy * particle_charge / particle_mass)
                / METERS_PER_KILOMETER
            )
        return (
            np.sqrt(2 * energy * particle_charge / particle_mass) / METERS_PER_KILOMETER
        )
    else:
        return (
            umath.sqrt(2 * energy * particle_charge / particle_mass)
            / METERS_PER_KILOMETER
        )


def read_mag_rtn_data(cdf_path) -> MagData:
    with CDF(str(cdf_path)) as cdf:
        var = cdf["b_rtn"]
        data = read_numeric_variable(var)[:, :3]
        attrs = var.attrs
        if "VALIDMIN" in attrs:
            data = np.where(data < float(attrs["VALIDMIN"]), np.nan, data)
        if "VALIDMAX" in attrs:
            data = np.where(data > float(attrs["VALIDMAX"]), np.nan, data)
        return MagData(
            epoch=pycdf.lib.v_datetime_to_tt2000(cdf["epoch"][...]),
            mag_data=data,
        )


def read_l2_swapi_data(cdf: CDF) -> SwapiL2Data:
    sci_start_times = pycdf.lib.v_datetime_to_tt2000(
        [datetime.fromisoformat(x) for x in cdf["sci_start_time"][...]]
    )
    return SwapiL2Data(
        sci_start_times,
        read_numeric_variable(cdf["esa_energy"]),
        read_numeric_variable(cdf["swp_coin_rate"]),
        read_numeric_variable(cdf["swp_coin_rate_stat_uncert_plus"]),
    )


def get_swapi_geometry(measurement_time: ndarray) -> ndarray:
    et_times = ttj2000ns_to_et(np.atleast_1d(measurement_time))
    return get_rotation_matrix(et_times, SpiceFrame.IMAP_SWAPI, SpiceFrame.IMAP_RTN)


def rotate_rtn_to_dps(vector_rtn, epoch_tt2000_ns: float):
    et = float(ttj2000ns_to_et(epoch_tt2000_ns))
    return frame_transform(
        et, np.asarray(vector_rtn), SpiceFrame.IMAP_RTN, SpiceFrame.IMAP_DPS
    )


def get_spacecraft_velocity_rtn(epoch_tt2000_ns: float) -> ndarray:
    et = float(ttj2000ns_to_et(epoch_tt2000_ns))
    state_eclipj2000 = imap_state(et, SpiceFrame.ECLIPJ2000)
    rtn_from_eclipj2000 = get_rotation_matrix(
        et, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN
    )
    return np.einsum("ij,j->i", rtn_from_eclipj2000, state_eclipj2000[3:])


def compute_direction_of_mean_magnetic_field_over_chunk(
    mag_data,
    chunk_epoch_center_tt2000_ns: int,
    chunk_epoch_delta_ns: int,
) -> np.ndarray:
    start = chunk_epoch_center_tt2000_ns - chunk_epoch_delta_ns
    end = chunk_epoch_center_tt2000_ns + chunk_epoch_delta_ns
    left = np.searchsorted(mag_data.epoch, start, side="left")
    right = np.searchsorted(mag_data.epoch, end, side="left")
    if right == left:
        return np.full(3, np.nan)
    b_mean = mag_data.mag_data[left:right].mean(axis=0)
    if not np.all(np.isfinite(b_mean)):
        return np.full(3, np.nan)
    return b_mean / np.linalg.norm(b_mean)


def chunk_l2_data(data: SwapiL2Data, chunk_size: int) -> Iterable[SwapiL2Data]:
    n = len(data.sci_start_time)
    for i in range(0, n - n % chunk_size, chunk_size):
        yield SwapiL2Data(
            data.sci_start_time[i : i + chunk_size],
            data.energy[i : i + chunk_size],
            data.coincidence_count_rate[i : i + chunk_size],
            data.coincidence_count_rate_uncertainty[i : i + chunk_size],
        )


def chunk_epoch(chunk: SwapiL2Data) -> float:
    return chunk.sci_start_time[0] + THIRTY_SECONDS_IN_NANOSECONDS


def measurement_times(chunk: SwapiL2Data, bin_slice: slice) -> ndarray:
    bins = np.arange(bin_slice.start, bin_slice.stop)
    return (
        chunk.sci_start_time[:, np.newaxis]
        + bins * (12 / 72 * ONE_SECOND_IN_NANOSECONDS)
    ).flatten()


def optimal_density_scale(unit_ideal_rates: ndarray, observed_rates: ndarray) -> float:
    def predicted_observed_rate(unit_rate, density):
        true_rate = density * unit_rate
        return true_rate * deadtime_factor(true_rate)

    popt, _ = scipy.optimize.curve_fit(
        f=predicted_observed_rate, xdata=unit_ideal_rates, ydata=observed_rates, p0=[1]
    )
    return float(popt[0])
