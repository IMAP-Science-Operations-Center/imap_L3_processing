"""Download one day of SWAPI L2 + dependencies, run the production alpha
solar-wind fit via SwapiProcessor, cache the result, and plot density /
temperature / bulk speed vs time.

Usage:
    scripts/swapi/fit_and_plot_alpha.py <YYYY-MM-DD> [--use-cache]

Requires the environment variable IMAP_API_KEY to be set.
Downloads land in /tmp/swapi_fit_and_plot_data; the fit pickle lands in
/tmp/swapi_alpha_fit_<YYYYMMDD>.pkl. --use-cache reuses that pickle and skips
all network access.
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from imap_data_access import ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input
from spacepy.pycdf import lib as cdf_library

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.swapi_processor import SwapiProcessor
from imap_l3_processing.utils import SpiceKernelTypes, get_spice_kernels_file_names

argument_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
argument_parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
argument_parser.add_argument("--use-cache", action="store_true",
                             help="Skip download and refit; load the cached fit from /tmp.")
arguments = argument_parser.parse_args()

if "IMAP_API_KEY" not in os.environ:
    sys.exit("IMAP_API_KEY environment variable is required.")

target_date = datetime.strptime(arguments.date, "%Y-%m-%d")
compact_date = target_date.strftime("%Y%m%d")
cache_path = Path(f"/tmp/swapi_alpha_fit_{compact_date}.pkl")

# Route the imap-data-access download cache to /tmp so we never write into the project tree.
imap_data_access.config["DATA_DIR"] = Path("/tmp/swapi_fit_and_plot_data")
imap_data_access.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

# Static list of SWAPI ancillaries; the same template scripts/swapi/generate_with_predicted_ephemeris.py uses.
ancillary_template_path = Path(__file__).parent / "imap_swapi_l3a_proton-sw_dependency_template.json"

if not arguments.use_cache or not cache_path.exists():
    # SWAPI L2 science for the target date.
    science_query_results = imap_data_access.query(
        instrument="swapi", data_level="l2", descriptor="sci",
        start_date=compact_date, end_date=compact_date, version="latest")
    if not science_query_results:
        sys.exit(f"No SWAPI L2 sci file at the SDC for {arguments.date}")

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
        sys.exit(f"No MAG norm-rtn file (L2 or L1D) at the SDC for {arguments.date}")

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

    # Per-day inputs (L2 + MAG + SPICE) plus the static ancillary template,
    # downloaded together via the production collection helpers.
    dynamic_filenames = [
        os.path.basename(science_query_results[0]["file_path"]),
        *magnetic_field_filenames,
        *spice_kernel_filenames,
    ]
    processing_input_collection = ProcessingInputCollection(
        *[generate_imap_input(filename) for filename in dynamic_filenames])
    processing_input_collection.deserialize(ancillary_template_path.read_text())
    processing_input_collection.download_all_files()

    # Furnish SPICE kernels before fitting — the alpha chunk fitter calls into
    # SPICE for per-measurement rotation matrices and spacecraft ephemeris.
    for kernel_path in processing_input_collection.get_file_paths(data_type="spice"):
        spiceypy.furnsh(str(imap_data_access.download(kernel_path)))

    # Run the production alpha-sw fit via SwapiProcessor.
    dependencies = SwapiL3ADependencies.fetch_dependencies(processing_input_collection)
    input_metadata = InputMetadata(
        instrument="swapi", data_level="l3a",
        start_date=target_date, end_date=target_date,
        version="v001", descriptor="alpha-sw",
    )
    processor = SwapiProcessor(processing_input_collection, input_metadata)
    alpha_solar_wind_data = processor.process_l3a_alpha(dependencies.data, dependencies)

    with cache_path.open("wb") as cache_file:
        pickle.dump(alpha_solar_wind_data, cache_file)
    print(f"Cached fit to {cache_path}")

with cache_path.open("rb") as cache_file:
    alpha_solar_wind_data = pickle.load(cache_file)

timestamps = np.array(
    [cdf_library.tt2000_to_datetime(int(tt2000))
     for tt2000 in alpha_solar_wind_data.epoch])

panels = [
    (alpha_solar_wind_data.alpha_sw_density,
     alpha_solar_wind_data.alpha_sw_density_uncert, "Density [cm$^{-3}$]", "log"),
    (alpha_solar_wind_data.alpha_sw_temperature,
     alpha_solar_wind_data.alpha_sw_temperature_uncert, "Temperature [K]", "log"),
    (alpha_solar_wind_data.alpha_sw_speed,
     alpha_solar_wind_data.alpha_sw_speed_uncert, "Bulk speed [km/s]", "linear"),
]
figure, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(10, 8))
for axis, (values, uncertainties, ylabel, yscale) in zip(axes, panels):
    if uncertainties is None:
        axis.plot(timestamps, values, ".")
    else:
        axis.errorbar(timestamps, values, yerr=uncertainties, fmt=".", capsize=2)
    axis.set_ylabel(ylabel)
    axis.set_yscale(yscale)
axes[-1].set_xlabel("Time (UTC)")
figure.suptitle(f"SWAPI alpha solar-wind moments — {arguments.date}")
figure.autofmt_xdate()
plt.tight_layout()
plt.show()
