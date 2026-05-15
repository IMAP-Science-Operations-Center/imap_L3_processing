"""View one chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production-model (replayed) and the xarray-reference forward-model rates.

Loads the cached pickle output of fit_and_plot_pui.py (observed and
production-model spectrograms, fit parameters, per-chunk SW velocity) and
evaluates the xarray reference model for the selected chunk against the same
SWAPI calibration the fit used. Both PUI forward models are plotted without
background (matches the production replay).

Usage:
    scripts/swapi/view_one_pui_spectrum.py <YYYY-MM-DD> <HH:MM[:SS]>

Selects the 50-sweep chunk whose central epoch is nearest to the requested
date and time. Run `fit_and_plot_pui.py <date>` first to populate the pickle
caches.
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

from imap_l3_processing.constants import ONE_AU_IN_KM, ONE_SECOND_IN_NANOSECONDS
from imap_l3_processing.swapi.constants import SWAPI_COARSE_SWEEP_BINS, SWAPI_L2_K_FACTOR
from imap_l3_processing.swapi.descriptors import (
    AZIMUTHAL_TRANSMISSION_DESCRIPTOR,
    CENTRAL_EFFECTIVE_AREA_DESCRIPTOR,
    DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR,
    PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.utils import (
    rotate_rtn_velocity_to_swapi_per_bin,
)
from imap_l3_processing.swapi.l3a.science.pickup_ion.vasyliunas_siscoe_distribution import (
    build_vasyliunas_siscoe_distribution,
)
from imap_l3_processing.swapi.l3a.swapi_l3a_dependencies import SwapiL3ADependencies
from imap_l3_processing.swapi.l3a.utils import chunk_l2_data
from imap_l3_processing.utils import SpiceKernelTypes, get_spice_kernels_file_names
from scripts.swapi.pui_xarray_reference_50sweep import (
    build_pui_xarray_context,
    evaluate_pui_sweep_xarray,
)


argument_parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
argument_parser.add_argument("date", help="UTC date in YYYY-MM-DD format")
argument_parser.add_argument("time",
                             help="UTC time in HH:MM[:SS] format; picks the 50-sweep "
                                  "chunk whose central epoch is nearest.")
argument_parser.add_argument("--output-path", type=Path, default=None,
                             help="If set, save the figure to this path and skip plt.show().")
arguments = argument_parser.parse_args()


def _parse_hh_mm_ss(time_string: str):
    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_string, time_format).time()
        except ValueError:
            continue
    raise SystemExit(f"Could not parse time {time_string!r}; expected HH:MM[:SS].")

if "IMAP_API_KEY" not in os.environ:
    sys.exit("IMAP_API_KEY environment variable is required.")

target_date = datetime.strptime(arguments.date, "%Y-%m-%d")
target_datetime_utc = datetime.combine(target_date.date(), _parse_hh_mm_ss(arguments.time))
compact_date = target_date.strftime("%Y%m%d")
fit_cache_path = Path(f"/tmp/swapi_pui_fit_{compact_date}.pkl")
spectrogram_cache_path = Path(
    f"/tmp/swapi_pui_spectrograms_per_sweep_{compact_date}.pkl")
if not fit_cache_path.exists() or not spectrogram_cache_path.exists():
    sys.exit(f"Missing pickle caches for {arguments.date}. "
             f"Run scripts/swapi/fit_and_plot_pui.py {arguments.date} first.")

with fit_cache_path.open("rb") as cache_file:
    pickup_ion_data, sw_velocity_rtn_by_chunk_epoch = pickle.load(cache_file)
with spectrogram_cache_path.open("rb") as spectrogram_file:
    (energies_per_sweep_ev, observed_spectrogram, model_spectrogram,
     _sweep_epochs_tt2000) = pickle.load(spectrogram_file)

imap_data_access.config["DATA_DIR"] = Path("/tmp/swapi_fit_and_plot_data")
imap_data_access.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

ancillary_template_path = (Path(__file__).parent
                           / "imap_swapi_l3a_proton-sw_dependency_template.json")
science_query_results = imap_data_access.query(
    instrument="swapi", data_level="l2", descriptor="sci",
    start_date=compact_date, end_date=compact_date, version="latest")
if not science_query_results:
    sys.exit(f"No SWAPI L2 sci file at the SDC for {arguments.date}")

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
    *spice_kernel_filenames,
]
processing_input_collection = ProcessingInputCollection(
    *[generate_imap_input(filename) for filename in dynamic_filenames])
processing_input_collection.deserialize(ancillary_template_path.read_text())
processing_input_collection.download_all_files()

for kernel_path in processing_input_collection.get_file_paths(data_type="spice"):
    spiceypy.furnsh(str(imap_data_access.download(kernel_path)))

dependencies = SwapiL3ADependencies.fetch_dependencies(processing_input_collection)


def _swapi_ancillary_path(descriptor: str) -> Path:
    paths = processing_input_collection.get_file_paths(
        source="swapi", descriptor=descriptor)
    return imap_data_access.download(paths[0])


pui_xarray_context = build_pui_xarray_context(
    azimuthal_transmission_path=_swapi_ancillary_path(AZIMUTHAL_TRANSMISSION_DESCRIPTOR),
    central_effective_area_path=_swapi_ancillary_path(CENTRAL_EFFECTIVE_AREA_DESCRIPTOR),
    passband_fit_coefficients_path=_swapi_ancillary_path(PASSBAND_FIT_COEFFICIENTS_DESCRIPTOR),
    density_of_neutral_helium_lut_path=_swapi_ancillary_path(DENSITY_OF_NEUTRAL_HELIUM_DESCRIPTOR),
)

chunks = list(chunk_l2_data(dependencies.data, 50))
chunk_central_datetimes = np.array([
    cdf_library.tt2000_to_datetime(int(t)) for t in pickup_ion_data.epoch
])
seconds_from_target = np.array([
    abs((central - target_datetime_utc).total_seconds())
    for central in chunk_central_datetimes
])
chunk_index = int(np.argmin(seconds_from_target))
data_chunk = chunks[chunk_index]
print(f"Selected chunk {chunk_index} (central epoch "
      f"{chunk_central_datetimes[chunk_index].isoformat()}, "
      f"{seconds_from_target[chunk_index]:.1f} s from requested "
      f"{target_datetime_utc.isoformat()})")

chunk_epoch_tt2000 = int(pickup_ion_data.epoch[chunk_index])
sw_velocity_rtn_kms = sw_velocity_rtn_by_chunk_epoch.get(chunk_epoch_tt2000)
if sw_velocity_rtn_kms is None or np.any(np.isnan(sw_velocity_rtn_kms)):
    sys.exit(f"No proton SW velocity available for chunk {chunk_index}.")
if not np.isfinite(pickup_ion_data.cooling_index[chunk_index].n):
    sys.exit(f"Chunk {chunk_index} has no successful PUI fit.")

chunk_ephemeris_time = spiceypy.unitim(
    chunk_epoch_tt2000 / ONE_SECOND_IN_NANOSECONDS, "TT", "ET")
vasyliunas_siscoe = build_vasyliunas_siscoe_distribution(
    chunk_ephemeris_time, sw_velocity_rtn_kms,
    dependencies.density_of_neutral_helium_calibration_table,
    dependencies.helium_inflow_vector,
)
bulk_sw_per_bin_swapi_kms = rotate_rtn_velocity_to_swapi_per_bin(
    data_chunk, sw_velocity_rtn_kms)

# Production averages per-sweep rates computed with the per-bin bulk SW
# vector. At high E the f_PUI azimuthal window is narrow (~tens of degrees),
# so the rate is strongly nonlinear in bulk_az and evaluating once at the
# chunk-mean bulk under-predicts vs. averaging per-sweep evaluations.
# evaluate_pui_sweep_xarray expects the per-step SW vector aligned with the
# xarray context's V axis (ascending). The L2 coarse-sweep voltage axis is
# descending, so reorder before passing.
l2_voltage_axis_v = data_chunk.energy[0, SWAPI_COARSE_SWEEP_BINS] / SWAPI_L2_K_FACTOR
ascending_voltage_order = np.argsort(l2_voltage_axis_v)

fit_cooling_index = float(pickup_ion_data.cooling_index[chunk_index].n)
fit_cutoff_speed_kms = float(pickup_ion_data.cutoff_speed[chunk_index].n)
fit_ionization_rate_hz = float(pickup_ion_data.ionization_rate[chunk_index].n)

xarray_rate_per_sweep = np.empty(bulk_sw_per_bin_swapi_kms.shape[:2])
n_sweeps_in_chunk = bulk_sw_per_bin_swapi_kms.shape[0]
print(f"Evaluating xarray reference per-sweep ({n_sweeps_in_chunk} sweeps)...")
for sweep_index_in_chunk in range(n_sweeps_in_chunk):
    sweep_bulk_sw_swapi_kms = (
        bulk_sw_per_bin_swapi_kms[sweep_index_in_chunk][ascending_voltage_order]
    )
    xarray_rate_per_sweep[sweep_index_in_chunk] = evaluate_pui_sweep_xarray(
        pui_xarray_context,
        sweep_bulk_sw_swapi_kms,
        cooling_index=fit_cooling_index,
        cutoff_speed_kms=fit_cutoff_speed_kms,
        ionization_rate_hz=fit_ionization_rate_hz,
        heliocentric_distance_au=vasyliunas_siscoe.distance_km / ONE_AU_IN_KM,
        inflow_psi_deg=vasyliunas_siscoe.psi,
        solar_wind_speed_inertial_kms=vasyliunas_siscoe.solar_wind_speed_inertial_frame,
    )
xarray_rate_per_step = xarray_rate_per_sweep.mean(axis=0)

chunk_sweep_slice = slice(chunk_index * 50, (chunk_index + 1) * 50)
chunk_observed_per_sweep = observed_spectrogram[chunk_sweep_slice]
chunk_model_per_sweep = model_spectrogram[chunk_sweep_slice]
chunk_voltages_per_sweep = energies_per_sweep_ev[chunk_sweep_slice] / SWAPI_L2_K_FACTOR

chunk_observed_mean = np.nanmean(chunk_observed_per_sweep, axis=0)
chunk_model_mean = np.nanmean(chunk_model_per_sweep, axis=0)
chunk_energies_mean_ev = np.nanmean(chunk_voltages_per_sweep, axis=0) * SWAPI_L2_K_FACTOR

xarray_energies_ev = pui_xarray_context.voltages_v * SWAPI_L2_K_FACTOR

figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
axis.plot(chunk_energies_mean_ev, chunk_observed_mean,
          ".-", color="tab:blue", label="Observed (chunk mean)")
axis.plot(chunk_energies_mean_ev, chunk_model_mean,
          "x-", color="tab:orange", label="Production model (replayed)")
axis.plot(xarray_energies_ev, xarray_rate_per_step,
          "+-", color="tab:green",
          label="xarray reference (per-sweep avg)")
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel("Energy / eV")
axis.set_ylabel("Coincidence rate [Hz]")
axis.set_title(
    f"SWAPI He+ PUI spectrum — {chunk_central_datetimes[chunk_index].isoformat()} UT"
    f" (chunk {chunk_index})\n"
    f"α={fit_cooling_index:.2f}  "
    f"v_b={fit_cutoff_speed_kms:.0f} km/s  "
    f"β_E={fit_ionization_rate_hz:.2e} 1/s")
axis.grid(True, which="both", alpha=0.3)
axis.legend()
if arguments.output_path is not None:
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_path, bbox_inches="tight")
    print(f"Saved {arguments.output_path}")
else:
    plt.show()
