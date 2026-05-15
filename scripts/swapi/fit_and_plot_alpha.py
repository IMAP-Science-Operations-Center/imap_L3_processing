"""Download one day of SWAPI L2 + dependencies, run the production alpha
solar-wind L3a CDF via ``imap_l3_data_processor.py`` (the same entry point
the integration test exercises), and plot density / temperature / bulk
speed / RTN velocity components vs time. WIND SWE 2-min alpha moments are
overplotted as scatter points for comparison.

Usage:
    scripts/swapi/fit_and_plot_alpha.py <YYYY-MM-DD> [--use-cache]

Requires the environment variable IMAP_API_KEY to be set.
Downloads and the produced CDF land under /tmp/swapi_fit_and_plot_data
(``IMAP_DATA_DIR``). --use-cache reuses the existing alpha-sw CDF if one
is already there and skips the subprocess.

WIND alphas come from
ftp://nssdcftp.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/wind_swe_2m_sw<YYYYMM>.asc
and use the L1 Sun-Earth-line approximation vR=-Vx_GSE, vT=-Vy_GSE, vN=+Vz_GSE.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
import matplotlib.pyplot as plt
import numpy as np
from imap_data_access import DependencyFilePath, ProcessingInputCollection, ScienceFilePath
from imap_data_access.processing_input import generate_imap_input
from spacepy.pycdf import CDF

import imap_l3_processing
from imap_l3_processing.cdf.cdf_utils import read_numeric_variable
from imap_l3_processing.utils import SpiceKernelTypes, get_spice_kernels_file_names

# m_p / (2 k_B) = 60.5685 K/(km/s)^2 for protons; alpha mass = 4 m_p.
WIND_ALPHA_TEMPERATURE_FACTOR_K_PER_KM2_PER_S2 = 4.0 * 60.5685
# Fill threshold: real plasma + position values in the SWE 2-min file stay
# well below 9000 in every numeric column.
WIND_FILL_THRESHOLD = 9000.0
WIND_FTP_URL_TEMPLATE = (
    "ftp://nssdcftp.gsfc.nasa.gov/pub/data/wind/swe/ascii/2-min/"
    "wind_swe_2m_sw{yyyymm}.asc"
)

L3_PROCESSING_ROOT = Path(imap_l3_processing.__file__).parent.parent
DATA_DIR = Path("/tmp/swapi_fit_and_plot_data")
PROCESSOR_OUTPUT_VERSION = "v001"


def fetch_wind_swe_2min_alphas(compact_date_string, cache_directory):
    """Return WIND SWE 2-min alpha moments for one UTC day, converted to RTN.

    Mirrors the helper in ~/projects/imap-validation/_lib.py. Returns None
    on network or parse failure (with a warning printed to stderr).
    """
    year_month_string = compact_date_string[:6]
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_file_path = cache_directory / f"wind_swe_2m_sw{year_month_string}.asc"

    if not cache_file_path.exists():
        url = WIND_FTP_URL_TEMPLATE.format(yyyymm=year_month_string)
        print(f"Downloading WIND SWE 2-min {year_month_string}...", end=" ", flush=True)
        try:
            subprocess.run(
                ["curl", "--ssl-reqd", "-fsS",
                 "--connect-timeout", "15", "--max-time", "300",
                 "-o", str(cache_file_path), url],
                check=True,
            )
            print("done")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"failed ({exc})", file=sys.stderr)
            if cache_file_path.exists() and cache_file_path.stat().st_size == 0:
                cache_file_path.unlink()
            return None

    raw_table = np.loadtxt(cache_file_path, comments=";")
    year_column = raw_table[:, 0].astype(int)
    fractional_day_of_year_column = raw_table[:, 1].astype(float)
    # Mask fills on plasma columns; year (col 0) and fit_flag (col 2) are
    # integer-valued and never approach the threshold.
    plasma_column_indices = [i for i in range(raw_table.shape[1]) if i not in (0, 2)]
    plasma_block = raw_table[:, plasma_column_indices]
    plasma_block[np.abs(plasma_block) >= WIND_FILL_THRESHOLD] = np.nan
    raw_table[:, plasma_column_indices] = plasma_block

    # FDOY convention: midnight Jan 1 = 1.0, noon Jan 1 = 1.5.
    year_starts_nanoseconds = {
        int(y): np.datetime64(f"{int(y):04d}-01-01", "ns").astype("int64")
        for y in np.unique(year_column)
    }
    year_start_nanoseconds_column = np.array(
        [year_starts_nanoseconds[int(y)] for y in year_column], dtype="int64")
    epoch_nanoseconds = year_start_nanoseconds_column + (
        (fractional_day_of_year_column - 1.0) * 86400.0 * 1e9).astype("int64")
    epoch = epoch_nanoseconds.view("datetime64[ns]")

    day_start = np.datetime64(
        f"{compact_date_string[:4]}-{compact_date_string[4:6]}-{compact_date_string[6:8]}", "ns")
    day_end = day_start + np.timedelta64(1, "D")
    day_mask = (epoch >= day_start) & (epoch < day_end)
    if not np.any(day_mask):
        print(f"No WIND SWE rows match {compact_date_string}", file=sys.stderr)
        return None

    row = raw_table[day_mask]
    epoch_day = epoch[day_mask]

    alpha_speed = row[:, 23]
    alpha_speed_sigma = row[:, 24]
    alpha_velocity_x_gse, alpha_velocity_x_gse_sigma = row[:, 25], row[:, 26]
    alpha_velocity_y_gse, alpha_velocity_y_gse_sigma = row[:, 27], row[:, 28]
    alpha_velocity_z_gse, alpha_velocity_z_gse_sigma = row[:, 29], row[:, 30]
    alpha_thermal_speed, alpha_thermal_speed_sigma = row[:, 31], row[:, 32]
    alpha_density = row[:, 37]
    alpha_density_sigma = row[:, 38]

    # T = c * W^2 -> sigma_T = 2 * c * W * sigma_W (linear propagation, W >= 0).
    alpha_temperature = WIND_ALPHA_TEMPERATURE_FACTOR_K_PER_KM2_PER_S2 * alpha_thermal_speed**2
    alpha_temperature_sigma = (
        2.0 * WIND_ALPHA_TEMPERATURE_FACTOR_K_PER_KM2_PER_S2
        * np.abs(alpha_thermal_speed) * alpha_thermal_speed_sigma)

    # GSE -> RTN at L1, Sun-Earth-line approximation.
    return {
        "epoch": epoch_day,
        "speed": alpha_speed,
        "speed_sigma": alpha_speed_sigma,
        "density": alpha_density,
        "density_sigma": alpha_density_sigma,
        "temperature": alpha_temperature,
        "temperature_sigma": alpha_temperature_sigma,
        "velocity_r": -alpha_velocity_x_gse,
        "velocity_r_sigma": alpha_velocity_x_gse_sigma,
        "velocity_t": -alpha_velocity_y_gse,
        "velocity_t_sigma": alpha_velocity_y_gse_sigma,
        "velocity_n": alpha_velocity_z_gse,
        "velocity_n_sigma": alpha_velocity_z_gse_sigma,
    }


def build_dependency_collection(target_date, compact_date):
    """Query SDC for one day of SWAPI L2 + MAG RTN + SPICE kernels and merge
    them with the static SWAPI ancillary template into a ProcessingInputCollection.
    """
    science_query_results = imap_data_access.query(
        instrument="swapi", data_level="l2", descriptor="sci",
        start_date=compact_date, end_date=compact_date, version="latest")
    if not science_query_results:
        sys.exit(f"No SWAPI L2 sci file at the SDC for {compact_date}")

    # MAG RTN at both L2 and L1D. SwapiL3ADependencies picks L2 when both are
    # present and flags the result as preliminary when only L1D is available.
    magnetic_field_filenames = [
        os.path.basename(row["file_path"])
        for level in ("l2", "l1d")
        for row in imap_data_access.query(
            instrument="mag", data_level=level, descriptor="norm-rtn",
            start_date=compact_date, end_date=compact_date, version="latest")
    ]
    if not magnetic_field_filenames:
        sys.exit(f"No MAG norm-rtn file (L2 or L1D) at the SDC for {compact_date}")

    # SPICE kernels in two windows. Non-ephemeris kernels come from a narrow
    # window around the target date. Spacecraft ephemeris kernels need a wide
    # lookback because the reconstructed kernel ends weeks before the target
    # date and the predicted kernel covers the gap onward.
    ephemeris_kernel_types = [SpiceKernelTypes.EphemerisReconstructed,
                              SpiceKernelTypes.EphemerisPredicted]
    non_ephemeris_kernel_types = [t for t in SpiceKernelTypes
                                  if t not in ephemeris_kernel_types]
    spice_kernel_filenames = list(dict.fromkeys(
        os.path.basename(path) for path in (
            get_spice_kernels_file_names(
                target_date - timedelta(days=1),
                target_date + timedelta(days=2),
                non_ephemeris_kernel_types)
            + get_spice_kernels_file_names(
                target_date - timedelta(days=90),
                target_date + timedelta(days=2),
                ephemeris_kernel_types)
        )
    ))

    dynamic_filenames = [
        os.path.basename(science_query_results[0]["file_path"]),
        *magnetic_field_filenames,
        *spice_kernel_filenames,
    ]
    ancillary_template_path = (
        Path(__file__).parent / "imap_swapi_l3a_proton-sw_dependency_template.json")
    processing_input_collection = ProcessingInputCollection(
        *[generate_imap_input(filename) for filename in dynamic_filenames])
    processing_input_collection.deserialize(ancillary_template_path.read_text())
    return processing_input_collection


argument_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
argument_parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
argument_parser.add_argument("--use-cache", action="store_true",
                             help="Skip download/processing if the alpha-sw CDF already exists in IMAP_DATA_DIR.")
arguments = argument_parser.parse_args()

if "IMAP_API_KEY" not in os.environ:
    sys.exit("IMAP_API_KEY environment variable is required.")

target_date = datetime.strptime(arguments.date, "%Y-%m-%d")
compact_date = target_date.strftime("%Y%m%d")

# Route imap-data-access cache to /tmp so we never write into the project tree.
# Set the env var so the child imap_l3_data_processor.py subprocess sees the same DATA_DIR.
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["IMAP_DATA_DIR"] = str(DATA_DIR)
imap_data_access.config["DATA_DIR"] = DATA_DIR

output_cdf_path = ScienceFilePath(
    f"imap_swapi_l3a_alpha-sw_{compact_date}_{PROCESSOR_OUTPUT_VERSION}.cdf").construct_path()

if not arguments.use_cache or not output_cdf_path.exists():
    processing_input_collection = build_dependency_collection(target_date, compact_date)
    processing_input_collection.download_all_files()

    dependency_filename = (
        f"imap_swapi_l3a_alpha-sw_{compact_date}_{PROCESSOR_OUTPUT_VERSION}.json")
    dependency_path = DependencyFilePath(dependency_filename).construct_path()
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_text(processing_input_collection.serialize())

    output_cdf_path.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, "imap_l3_data_processor.py",
         "--instrument", "swapi",
         "--data-level", "l3a",
         "--descriptor", "alpha-sw",
         "--start-date", compact_date,
         "--version", PROCESSOR_OUTPUT_VERSION,
         "--dependency", dependency_filename],
        cwd=L3_PROCESSING_ROOT,
    )
    if result.returncode != 0:
        sys.exit(f"imap_l3_data_processor.py exited with status {result.returncode}")
    if not output_cdf_path.exists():
        sys.exit(f"Processor finished but {output_cdf_path} is missing.")

with CDF(str(output_cdf_path)) as alpha_cdf:
    timestamps = alpha_cdf["epoch"][...]
    quality_flags = alpha_cdf["swp_flags"][...]
    swapi_density = read_numeric_variable(alpha_cdf["alpha_sw_density"])
    swapi_density_uncertainty = read_numeric_variable(alpha_cdf["alpha_sw_density_uncert"])
    swapi_temperature = read_numeric_variable(alpha_cdf["alpha_sw_temperature"])
    swapi_temperature_uncertainty = read_numeric_variable(alpha_cdf["alpha_sw_temperature_uncert"])
    swapi_speed = read_numeric_variable(alpha_cdf["alpha_sw_speed"])
    swapi_speed_uncertainty = read_numeric_variable(alpha_cdf["alpha_sw_speed_uncert"])
    swapi_velocity_rtn = read_numeric_variable(alpha_cdf["alpha_sw_velocity_rtn_sun"])
    swapi_velocity_rtn_covariance = read_numeric_variable(
        alpha_cdf["alpha_sw_velocity_rtn_covariance"])

# Drop flagged records so fill values don't drag the y-axes.
good_record_mask = quality_flags == 0
timestamps = timestamps[good_record_mask]
swapi_density = swapi_density[good_record_mask]
swapi_density_uncertainty = swapi_density_uncertainty[good_record_mask]
swapi_temperature = swapi_temperature[good_record_mask]
swapi_temperature_uncertainty = swapi_temperature_uncertainty[good_record_mask]
swapi_speed = swapi_speed[good_record_mask]
swapi_speed_uncertainty = swapi_speed_uncertainty[good_record_mask]
swapi_velocity_rtn = swapi_velocity_rtn[good_record_mask]
swapi_velocity_rtn_covariance = swapi_velocity_rtn_covariance[good_record_mask]

# Diagonal of the RTN velocity covariance gives variances on (vR, vT, vN).
swapi_velocity_rtn_sigma = np.sqrt(np.diagonal(
    swapi_velocity_rtn_covariance, axis1=1, axis2=2))

wind_alphas = fetch_wind_swe_2min_alphas(compact_date, DATA_DIR / "wind") or {}

panels = [
    ("Density [cm$^{-3}$]", "log",
     swapi_density, swapi_density_uncertainty,
     wind_alphas.get("density"), wind_alphas.get("density_sigma")),
    ("Temperature [K]", "log",
     swapi_temperature, swapi_temperature_uncertainty,
     wind_alphas.get("temperature"), wind_alphas.get("temperature_sigma")),
    ("Bulk speed [km/s]", "linear",
     swapi_speed, swapi_speed_uncertainty,
     wind_alphas.get("speed"), wind_alphas.get("speed_sigma")),
    ("$v_R$ [km/s]", "linear",
     swapi_velocity_rtn[:, 0], swapi_velocity_rtn_sigma[:, 0],
     wind_alphas.get("velocity_r"), wind_alphas.get("velocity_r_sigma")),
    ("$v_T$ [km/s]", "linear",
     swapi_velocity_rtn[:, 1], swapi_velocity_rtn_sigma[:, 1],
     wind_alphas.get("velocity_t"), wind_alphas.get("velocity_t_sigma")),
    ("$v_N$ [km/s]", "linear",
     swapi_velocity_rtn[:, 2], swapi_velocity_rtn_sigma[:, 2],
     wind_alphas.get("velocity_n"), wind_alphas.get("velocity_n_sigma")),
]
wind_epoch = wind_alphas.get("epoch")

figure, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(10, 14))
for axis, (ylabel, yscale, swapi_values, swapi_uncertainties,
           wind_values, wind_uncertainties) in zip(axes, panels):
    axis.errorbar(timestamps, swapi_values, yerr=swapi_uncertainties,
                  fmt=".", capsize=2, color="tab:blue", label="SWAPI L3a")
    if wind_values is not None and wind_epoch is not None:
        axis.errorbar(wind_epoch, wind_values, yerr=wind_uncertainties,
                      fmt=".", capsize=2, color="tab:orange", alpha=0.6,
                      label="WIND SWE 2-min")
    axis.set_ylabel(ylabel)
    axis.set_yscale(yscale)
axes[0].legend(loc="best")
axes[-1].set_xlabel("Time (UTC)")
figure.suptitle(f"SWAPI alpha solar-wind moments — {arguments.date}")
figure.autofmt_xdate()
plt.tight_layout()
plt.show()
