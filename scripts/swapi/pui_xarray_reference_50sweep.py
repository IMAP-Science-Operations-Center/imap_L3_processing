r"""Generate a realistic 50-sweep SWAPI coincidence-rate dataset.
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import pint
import pint_xarray  # noqa: F401
import spiceypy
import xarray as xr

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.constants import SWAPI_L2_K_FACTOR
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.utils import SpiceKernelTypes, furnish_spice_metakernel
from tests.swapi._helpers import load_swapi_response
from tests.test_helpers import get_test_data_path, get_test_instrument_team_data_path

OUTPUT_PATH = get_test_data_path("swapi/pui_count_rate_reference_50sweep.h5")

if "IMAP_API_KEY" not in os.environ:
    sys.exit(
        "IMAP_API_KEY environment variable is required."
    )

ureg = pint.UnitRegistry(force_ndarray_like=True)
ureg.define("counts = []")
pint.set_application_registry(ureg)
Q = ureg.Quantity

K_FACTOR = Q(1.89, "eV/V/e")
ONE_AU = Q(1.0, "au")
HE_PLUS_MASS_PER_CHARGE = Q(4.0, "m_p/e")
HELIUM_EFFICIENCY_RATIO = 1.05

# 50-sweep chunk over 10 minutes. Picks a date covered by the SPICE kernel
# release; the calibration CSVs are stamped 20260425 so we sit in that window.
START_TIME_UTC = datetime(2026, 4, 25, 0, 0, 0)
N_SWEEPS = 50
N_ESA_STEPS_PER_SWEEP = 62
SWEEP_DURATION_S = 12.0
# Coarse-sweep step cadence: a SWAPI sweep has 72 steps in 12 s; the 62 we
# care about take ~62/72 × 12 s. Spread the per-bin timestamps evenly across
# that window so the spin phase advances across the steps.
TOTAL_COARSE_DURATION_S = SWEEP_DURATION_S * N_ESA_STEPS_PER_SWEEP / 72
STEP_DURATION_S = TOTAL_COARSE_DURATION_S / N_ESA_STEPS_PER_SWEEP
END_TIME_UTC = START_TIME_UTC + timedelta(
    seconds=SWEEP_DURATION_S * N_SWEEPS + 60
)

# Truth values are chosen well inside every fit bound so frozen-value
# integration tests are not sensitive to LM termination next to a wall:
#   cooling_index ∈ [1.0, 5.0]           → 2.5 (centered)
#   ionization_rate ∈ [0.6e-9, 8e-7]     → 1e-7 (comfortable margin from both ends)
#   cutoff_speed ∈ [0.8·sw, 1.2·sw]      → 1.05·sw_speed (clear of both edges)
#   background_count_rate cap at 1.0 Hz  → 0.3 Hz (no post-fit fill)
SW_SPEED_KMS = 450.0
COOLING_INDEX = 2.5
CUTOFF_SPEED_KMS = 1.05 * SW_SPEED_KMS
IONIZATION_RATE_HZ = 1e-7
HELIO_DIST_AU = 1.0
INFLOW_PSI_DEG = 75.0
SW_SPEED_INERTIAL_KMS = 450.0
BACKGROUND_RATE_HZ = 0.3
HELIUM_MASS_PER_CHARGE_M_P_PER_E = 4.0

# Proton + alpha Maxwellian shoulder truth.
PROTON_DENSITY_CM3 = 5.0
PROTON_TEMPERATURE_K = 1.0e5
ALPHA_DENSITY_CM3 = 0.2
ALPHA_TEMPERATURE_K = 4.0e5

# Bulk SW velocity in IMAP_RTN: along +R (radially outward from the Sun).
BULK_SW_RTN_KMS = np.array([SW_SPEED_KMS, 0.0, 0.0])

print(f"Downloading SPICE kernels for {START_TIME_UTC} → {END_TIME_UTC}...")
furnish_spice_metakernel(
    start_date=START_TIME_UTC,
    end_date=END_TIME_UTC,
    kernel_types=[
        SpiceKernelTypes.Leapseconds,
        SpiceKernelTypes.SpacecraftClock,
        SpiceKernelTypes.IMAPFrames,
        SpiceKernelTypes.ScienceFrames,
        SpiceKernelTypes.AttitudeHistory,
        SpiceKernelTypes.PointingAttitude,
        SpiceKernelTypes.EphemerisReconstructed,
        SpiceKernelTypes.PlanetaryEphemeris,
        SpiceKernelTypes.PlanetaryConstants,
    ],
)

start_et = spiceypy.str2et(START_TIME_UTC.isoformat())
sweep_starts_et = start_et + np.arange(N_SWEEPS) * SWEEP_DURATION_S
step_offsets_s = np.arange(N_ESA_STEPS_PER_SWEEP) * STEP_DURATION_S
ephemeris_time_grid = (
    sweep_starts_et[:, None] + step_offsets_s[None, :]
)  # (sweep, step)

print("Computing per-bin IMAP_RTN↔IMAP_SWAPI rotations...")
bulk_sw_per_bin_swapi_kms = np.empty((N_SWEEPS, N_ESA_STEPS_PER_SWEEP, 3))
swapi_to_rtn_rotation_per_bin = np.empty((N_SWEEPS, N_ESA_STEPS_PER_SWEEP, 3, 3))
for sweep_index in range(N_SWEEPS):
    for step_index in range(N_ESA_STEPS_PER_SWEEP):
        rotation_imap_rtn_to_imap_swapi = spiceypy.pxform(
            "IMAP_RTN",
            "IMAP_SWAPI",
            float(ephemeris_time_grid[sweep_index, step_index]),
        )
        bulk_sw_per_bin_swapi_kms[sweep_index, step_index] = (
            rotation_imap_rtn_to_imap_swapi @ BULK_SW_RTN_KMS
        )
        swapi_to_rtn_rotation_per_bin[sweep_index, step_index] = (
            rotation_imap_rtn_to_imap_swapi.T
        )

# Calibration grids (same as pui_xarray_reference.py).
central_effective_area = HELIUM_EFFICIENCY_RATIO * (
    pd.read_csv(
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_central-effective-area_20260425_v001.csv"
        )
    )
    .set_index("esa_voltage")
    .effective_area.to_xarray()
    .rename(esa_voltage="V")
    .pint.quantify("cm**2")
)
_voltages_mag = central_effective_area["V"].values
voltages = xr.DataArray(
    Q(_voltages_mag, "V"), dims="V", coords={"V": _voltages_mag},
)

if voltages.sizes["V"] != N_ESA_STEPS_PER_SWEEP:
    raise RuntimeError(
        f"calibration voltage axis has {voltages.sizes['V']} entries, "
        f"expected {N_ESA_STEPS_PER_SWEEP} to match the coarse-sweep step count"
    )

passband_coefficients = (
    pd.read_csv(
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_passband-fit-coefficients_20260425_v001.csv"
        )
    )
    .set_index(["region", "energy_ratio", "elevation"])[["0", "1", "2"]]
    .rename_axis(columns="degree")
    .to_xarray()
    .to_dataarray("degree")
    .rename(elevation="elevation_deg")
    .pipe(lambda da: da.assign_coords(degree=da["degree"].astype(int)))
)
source_speed_ratio = np.sqrt(
    passband_coefficients["energy_ratio"].values / K_FACTOR.to("eV/V/e").magnitude
)
passband_coefficients = (
    passband_coefficients.assign_coords(
        speed_ratio=("energy_ratio", source_speed_ratio)
    )
    .swap_dims({"energy_ratio": "speed_ratio"})
    .drop_vars("energy_ratio")
)

az_transmission_native = (
    pd.read_csv(
        get_test_instrument_team_data_path(
            "swapi/imap_swapi_azimuthal-transmission_20260425_v001.csv"
        )
    )
    .fillna(0)
    .set_index("abs_azimuth")
    .transmission.to_xarray()
).coarsen(abs_azimuth=10, boundary="trim").mean()

density_data = np.loadtxt(
    get_test_instrument_team_data_path("swapi/density-of-neutral-helium-lut.dat")
)
psi_axis, r_axis = np.unique(density_data[:, 0]), np.unique(density_data[:, 1])
density_table = xr.DataArray(
    density_data[:, 2].reshape(len(psi_axis), len(r_axis)),
    dims=("psi", "r"),
    coords={"psi": psi_axis, "r": r_axis},
).pint.quantify("1/cm**3")

elevation_deg = passband_coefficients["elevation_deg"]
speed_ratio = passband_coefficients["speed_ratio"]

# Mirror signed [-180, 180] azimuth grid.
_abs_az = az_transmission_native["abs_azimuth"].values
_trans = az_transmission_native.values
_az_signed = np.concatenate([-_abs_az[:0:-1], _abs_az])
azimuth_deg = xr.DataArray(
    _az_signed, dims="azimuth_deg", coords={"azimuth_deg": _az_signed}
)
azimuthal_transmission = xr.DataArray(
    np.concatenate([_trans[:0:-1], _trans]),
    dims="azimuth_deg",
    coords={"azimuth_deg": _az_signed},
)

azimuth_rad = np.deg2rad(azimuth_deg)
elevation_rad = np.deg2rad(elevation_deg)
direction = xr.concat(
    [
        -np.cos(elevation_rad) * np.sin(azimuth_rad),
        -np.cos(elevation_rad) * np.cos(azimuth_rad),
        -np.sin(elevation_rad) * xr.ones_like(azimuth_rad),
    ],
    dim="cartesian",
).assign_coords(cartesian=["x", "y", "z"])

central_instrument_speed = np.sqrt(
    2 * K_FACTOR * voltages / HE_PLUS_MASS_PER_CHARGE
).pint.to("km/s")

log_beam_energy = np.log(
    np.abs(K_FACTOR * voltages).pint.to("eV/e").pint.dequantify()
)
passband_per_region = np.exp(
    xr.polyval(log_beam_energy, passband_coefficients, degree_dim="degree")
).fillna(0)
passband_per_region = passband_per_region / passband_per_region.interp(
    elevation_deg=0.0, speed_ratio=1.0,
)
az_is_sg = np.abs(azimuth_deg) <= 20
passband_full = xr.where(
    az_is_sg,
    passband_per_region.sel(region="SG"),
    passband_per_region.sel(region="OA"),
)

effective_area = (
    central_effective_area * passband_full * azimuthal_transmission
).clip(min=Q(0, "cm**2"))

speed_grid = (speed_ratio * central_instrument_speed).transpose("V", "speed_ratio")

cooling_index = COOLING_INDEX
cutoff_speed = Q(CUTOFF_SPEED_KMS, "km/s")
r = Q(HELIO_DIST_AU, "au")
u_sw = Q(SW_SPEED_INERTIAL_KMS, "km/s")
beta_E = Q(IONIZATION_RATE_HZ, "1/s")
psi = Q(INFLOW_PSI_DEG, "deg")

term1 = cooling_index / (4 * np.pi)
term2 = (beta_E * ONE_AU**2) / (r * u_sw * cutoff_speed**3)

deg2_to_sr = (np.pi / 180.0) ** 2
cos_elevation = np.cos(np.deg2rad(elevation_deg))


def evaluate_sweep(bulk_sw_per_step_kms: np.ndarray) -> np.ndarray:
    """Evaluate per-V coincidence rate (Hz) for one sweep.

    `bulk_sw_per_step_kms` has shape (N_V, 3) — the per-step bulk SW vector in
    SWAPI Cartesian. We build a per-V v_sw DataArray, recompute the SW-frame
    speed grid, evaluate the V-S distribution on it, and integrate over
    (azimuth, elevation, speed_ratio) to collapse to a (V,) coincidence rate.
    """
    v_sw = xr.DataArray(
        Q(bulk_sw_per_step_kms, "km/s"),
        dims=("V", "cartesian"),
        coords={"V": _voltages_mag, "cartesian": ["x", "y", "z"]},
    )
    v_sw_speed = np.sqrt((v_sw**2).sum("cartesian"))  # (V,)
    v_dot_vsw = xr.dot(direction, v_sw, dim="cartesian")  # (V, az, el)
    speed_sw = np.sqrt(
        np.maximum(
            speed_grid**2 + v_sw_speed**2 - 2 * speed_grid * v_dot_vsw,
            Q(0, "km**2/s**2"),
        )
    )

    w = (speed_sw / cutoff_speed).pint.to("dimensionless")
    term3 = w ** (cooling_index - 3)
    term4 = (
        density_table.pint.interp(
            psi=(psi % Q(360, "deg")).magnitude,
            r=(r * w**cooling_index).pint.to("au").pint.dequantify(),
        )
        .drop_vars(["psi", "r"])
        .fillna(0)
    )
    term5 = xr.where((w < 1), 1, 0)
    f_pui = (term1 * term2 * term3 * term4 * term5).pint.to("s**3/km**6")

    flux = f_pui * speed_grid
    d3v = speed_grid**2 * deg2_to_sr * cos_elevation
    integrand = effective_area * flux * d3v
    integral = (
        integrand.integrate("azimuth_deg")
        .integrate("elevation_deg")
        .integrate("speed_ratio")
    )
    count_rate = (integral * central_instrument_speed).pint.to("counts/s")
    return count_rate.pint.dequantify().values  # (V,)


pui_count_rate_per_sweep = np.empty((N_SWEEPS, N_ESA_STEPS_PER_SWEEP))
for sweep_index in range(N_SWEEPS):
    print(f"  PUI sweep {sweep_index + 1}/{N_SWEEPS}")
    pui_count_rate_per_sweep[sweep_index] = evaluate_sweep(
        bulk_sw_per_bin_swapi_kms[sweep_index]
    )

print("Computing proton + alpha Maxwellian shoulder via production forward model...")
voltage_repeated = np.broadcast_to(
    _voltages_mag, (N_SWEEPS, N_ESA_STEPS_PER_SWEEP)
).ravel()
rotation_flat = swapi_to_rtn_rotation_per_bin.reshape(-1, 3, 3)
swapi_response = load_swapi_response(warm_cache_voltages=_voltages_mag)

proton_truth = SolarWindParams(
    density=PROTON_DENSITY_CM3,
    bulk_velocity_rtn=BULK_SW_RTN_KMS.copy(),
    temperature=PROTON_TEMPERATURE_K,
    mass=PROTON_MASS_KG,
)
alpha_truth = SolarWindParams(
    density=ALPHA_DENSITY_CM3,
    bulk_velocity_rtn=BULK_SW_RTN_KMS.copy(),
    temperature=ALPHA_TEMPERATURE_K,
    mass=ALPHA_PARTICLE_MASS_KG,
)
proton_ctx = build_solar_wind_fit_context(
    count_rate=np.zeros(voltage_repeated.size),
    esa_voltage=voltage_repeated,
    swapi_response=swapi_response,
    central_effective_area_scale=1.0,
    rotation_matrices=rotation_flat,
    mass_kg=PROTON_MASS_KG,
    mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
alpha_ctx = build_solar_wind_fit_context(
    count_rate=np.zeros(voltage_repeated.size),
    esa_voltage=voltage_repeated,
    swapi_response=swapi_response,
    central_effective_area_scale=HELIUM_EFFICIENCY_RATIO,
    rotation_matrices=rotation_flat,
    mass_kg=ALPHA_PARTICLE_MASS_KG,
    mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
)
proton_ideal, _ = model_solar_wind_ideal_coincidence_rates(proton_truth, proton_ctx)
alpha_ideal, _ = model_solar_wind_ideal_coincidence_rates(alpha_truth, alpha_ctx)
proton_alpha_count_rate_per_sweep = (proton_ideal + alpha_ideal).reshape(
    N_SWEEPS, N_ESA_STEPS_PER_SWEEP
)

ideal_total_rate = (
    pui_count_rate_per_sweep
    + proton_alpha_count_rate_per_sweep
    + BACKGROUND_RATE_HZ
)
expected_count_rate_per_sweep = ideal_total_rate * deadtime_factor(ideal_total_rate)

# Per-sweep TT2000 timestamps (nanoseconds from J2000.0 TT epoch).
sci_start_time_tt2000_ns = np.array(
    [int(spiceypy.unitim(et, "ET", "TT") * 1e9) for et in sweep_starts_et],
    dtype=np.int64,
)

# ESA energy per step: voltage * L2 k-factor.
energy_ev = _voltages_mag * SWAPI_L2_K_FACTOR

print(f"Writing {OUTPUT_PATH}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(OUTPUT_PATH, "w") as h5:
    h5.create_dataset(
        "voltage_v",
        data=_voltages_mag,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "expected_coincidence_rate_hz",
        data=expected_count_rate_per_sweep,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "pui_coincidence_rate_hz",
        data=pui_count_rate_per_sweep,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "proton_alpha_coincidence_rate_hz",
        data=proton_alpha_count_rate_per_sweep,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "bulk_sw_per_bin_swapi_kms",
        data=bulk_sw_per_bin_swapi_kms,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "swapi_to_rtn_rotation_per_bin",
        data=swapi_to_rtn_rotation_per_bin,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "ephemeris_time_s",
        data=ephemeris_time_grid,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "sci_start_time_tt2000_ns",
        data=sci_start_time_tt2000_ns,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.create_dataset(
        "energy_ev",
        data=energy_ev,
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )
    h5.attrs["start_time_utc"] = START_TIME_UTC.isoformat()
    h5.attrs["n_sweeps"] = N_SWEEPS
    h5.attrs["n_esa_steps_per_sweep"] = N_ESA_STEPS_PER_SWEEP
    h5.attrs["sweep_duration_s"] = SWEEP_DURATION_S
    h5.attrs["bulk_sw_frame"] = "IMAP_RTN"
    h5.attrs["bulk_sw_rtn_kms"] = BULK_SW_RTN_KMS
    h5.attrs["sw_speed_kms"] = SW_SPEED_KMS
    h5.attrs["sw_speed_inertial_kms"] = SW_SPEED_INERTIAL_KMS
    h5.attrs["cooling_index"] = COOLING_INDEX
    h5.attrs["cutoff_speed_kms"] = CUTOFF_SPEED_KMS
    h5.attrs["ionization_rate_hz"] = IONIZATION_RATE_HZ
    h5.attrs["helio_dist_au"] = HELIO_DIST_AU
    h5.attrs["inflow_psi_deg"] = INFLOW_PSI_DEG
    h5.attrs["helium_efficiency_ratio"] = HELIUM_EFFICIENCY_RATIO
    h5.attrs["helium_mass_per_charge_m_p_per_e"] = HELIUM_MASS_PER_CHARGE_M_P_PER_E
    h5.attrs["background_rate_hz"] = BACKGROUND_RATE_HZ
    h5.attrs["proton_density_cm3"] = PROTON_DENSITY_CM3
    h5.attrs["proton_temperature_k"] = PROTON_TEMPERATURE_K
    h5.attrs["alpha_density_cm3"] = ALPHA_DENSITY_CM3
    h5.attrs["alpha_temperature_k"] = ALPHA_TEMPERATURE_K
print("done.")
