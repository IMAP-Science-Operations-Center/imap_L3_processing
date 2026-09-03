"""View one chunk's mean He+ PUI spectrum: observed coincidence rate vs the
production-model (replayed) and the xarray-reference forward-model rates.

Loads the cached pickle output of fit_and_plot_pui.py (observed and
production-model spectrograms, fit parameters, per-chunk SW velocity) and
evaluates the xarray reference model for the selected chunk against the same
SWAPI calibration the fit used. Both PUI forward models are plotted with the
fitted background count rate added, so they are directly comparable to the
observed coincidence rate.

Usage:
    scripts/swapi/view_one_pui_spectrum.py <YYYY-MM-DD> <HH:MM[:SS]> [--offline]

Selects the 50-sweep chunk whose central epoch is nearest to the requested
date and time. Run `fit_and_plot_pui.py <date>` first to populate the pickle
caches. Pass --offline to resolve the L2 file and SPICE kernels from the local
DATA_DIR cache (populated by a prior online run) instead of contacting the SDC.
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
from matplotlib.colors import LogNorm
from imap_data_access import ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input
from spacepy.pycdf import lib as cdf_library

from imap_processing.swapi.l2 import swapi_l2

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
argument_parser.add_argument("--offline", action="store_true",
                             help="Resolve the L2 file and SPICE kernels from the local "
                                  "DATA_DIR cache instead of querying/downloading from the "
                                  "SDC. Requires a prior online run to have populated the cache.")
arguments = argument_parser.parse_args()


def _parse_hh_mm_ss(time_string: str):
    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_string, time_format).time()
        except ValueError:
            continue
    raise SystemExit(f"Could not parse time {time_string!r}; expected HH:MM[:SS].")

if not arguments.offline and "IMAP_API_KEY" not in os.environ:
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
data_dir = imap_data_access.config["DATA_DIR"]


def resolve_l2_filename_from_cache() -> str:
    matches = sorted((data_dir / "imap" / "swapi" / "l2").rglob(
        f"imap_swapi_l2_sci_{compact_date}_v*.cdf"))
    if not matches:
        sys.exit(f"No cached SWAPI L2 sci file for {arguments.date} under {data_dir}. "
                 f"Run once without --offline to populate the cache.")
    return matches[-1].name


def resolve_spice_filenames_from_cache() -> list[str]:
    names = sorted(p.name for p in (data_dir / "imap" / "spice").rglob("*") if p.is_file())
    if not names:
        sys.exit(f"No cached SPICE kernels under {data_dir}. "
                 f"Run once without --offline to populate the cache.")
    return names


if arguments.offline:
    l2_science_filename = resolve_l2_filename_from_cache()
    spice_kernel_filenames = resolve_spice_filenames_from_cache()
else:
    science_query_results = imap_data_access.query(
        instrument="swapi", data_level="l2", descriptor="sci",
        start_date=compact_date, end_date=compact_date, version="latest")
    if not science_query_results:
        sys.exit(f"No SWAPI L2 sci file at the SDC for {arguments.date}")
    l2_science_filename = os.path.basename(science_query_results[0]["file_path"])

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

dynamic_filenames = [l2_science_filename, *spice_kernel_filenames]
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
fit_background_rate_hz = float(pickup_ion_data.background_rate[chunk_index].n)
# A rejected background is stored as NaN (see _set_background_to_fill_if_too_high);
# adding it would wipe out the model lines, so fall back to no offset.
background_offset_hz = fit_background_rate_hz if np.isfinite(fit_background_rate_hz) else 0.0

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

# Poisson uncertainty on the chunk-mean rate at each bin: the mean of N
# independent counting measurements over a livetime each, so total counts
# = N · mean_rate · livetime and σ(mean_rate) = sqrt(mean_rate / (N · livetime)).
valid_sweeps_per_bin = np.sum(np.isfinite(chunk_observed_per_sweep), axis=0)
chunk_observed_uncertainty = np.sqrt(
    chunk_observed_mean / (valid_sweeps_per_bin * swapi_l2.SWAPI_LIVETIME))

pickup_ion_window_bin_mask = ~np.all(np.isnan(chunk_model_per_sweep), axis=0)

xarray_energies_ev = pui_xarray_context.voltages_v * SWAPI_L2_K_FACTOR

mean_energies_ev_per_step = np.nanmean(
    chunk_voltages_per_sweep * SWAPI_L2_K_FACTOR, axis=0)
pickup_ion_bin_indices = np.where(pickup_ion_window_bin_mask)[0]
fit_window_energies_ev = mean_energies_ev_per_step[pickup_ion_bin_indices]
energy_sort_order_within_window = np.argsort(fit_window_energies_ev)
mean_energies_ev_sorted = fit_window_energies_ev[energy_sort_order_within_window]
sorted_pickup_ion_bin_indices = pickup_ion_bin_indices[energy_sort_order_within_window]
observed_spectrogram_sorted = chunk_observed_per_sweep[:, sorted_pickup_ion_bin_indices]
model_spectrogram_sorted = chunk_model_per_sweep[:, sorted_pickup_ion_bin_indices]

positive_spectrogram_values = np.concatenate([
    array[np.isfinite(array) & (array > 0)]
    for array in (observed_spectrogram_sorted, model_spectrogram_sorted)
])
if positive_spectrogram_values.size:
    spectrogram_vmin = max(
        float(np.percentile(positive_spectrogram_values, 1)), 1e-3)
    spectrogram_vmax = float(positive_spectrogram_values.max())
else:
    spectrogram_vmin, spectrogram_vmax = 1e-3, 1.0
spectrogram_norm = LogNorm(vmin=spectrogram_vmin, vmax=spectrogram_vmax)

figure = plt.figure(figsize=(10, 9), constrained_layout=True)
grid_spec = figure.add_gridspec(3, 1, height_ratios=[1.3, 0.8, 0.8])
line_axis = figure.add_subplot(grid_spec[0])
observed_spectrogram_axis = figure.add_subplot(grid_spec[1])
model_spectrogram_axis = figure.add_subplot(
    grid_spec[2], sharex=observed_spectrogram_axis,
    sharey=observed_spectrogram_axis)

line_axis.errorbar(chunk_energies_mean_ev, chunk_observed_mean,
                   yerr=chunk_observed_uncertainty,
                   fmt=".", color="tab:blue", elinewidth=0.8, capsize=2,
                   label="Observed (chunk mean)")
line_axis.plot(chunk_energies_mean_ev, chunk_model_mean + background_offset_hz,
               "x-", color="tab:orange", label="Model (Optimized) + background")
line_axis.plot(xarray_energies_ev, xarray_rate_per_step + background_offset_hz,
               "+-", color="tab:green", label="Model (Reference) + background")
line_axis.set_xscale("log")
line_axis.set_yscale("log")
line_axis.set_xlabel("Energy / eV")
line_axis.set_ylabel("Coincidence rate [Hz]")
line_axis.set_title(
    f"SWAPI He+ PUI spectrum — {chunk_central_datetimes[chunk_index].isoformat()} UT"
    f" (chunk {chunk_index})\n"
    f"α={fit_cooling_index:.2f}  "
    f"v_b={fit_cutoff_speed_kms:.0f} km/s  "
    f"β_E={fit_ionization_rate_hz:.2e} 1/s  "
    + (f"bg={fit_background_rate_hz:.3f} Hz"
       if np.isfinite(fit_background_rate_hz) else "bg=nan (rejected)"))
line_axis.grid(True, which="both", alpha=0.3)
line_axis.legend()

sweep_indices_in_chunk = np.arange(n_sweeps_in_chunk)
for axis_for_spectrogram, spectrogram_values, label in (
    (observed_spectrogram_axis, observed_spectrogram_sorted,
     "Observed (50 sweeps × PUI fit window)"),
    (model_spectrogram_axis, model_spectrogram_sorted, "Model"),
):
    mesh = axis_for_spectrogram.pcolormesh(
        sweep_indices_in_chunk, mean_energies_ev_sorted,
        np.ma.masked_invalid(spectrogram_values).T,
        shading="nearest", cmap="viridis", norm=spectrogram_norm,
        rasterized=True,
    )
    axis_for_spectrogram.set_yscale("log")
    axis_for_spectrogram.set_ylabel("Energy / eV")
    axis_for_spectrogram.text(
        0.01, 0.95, label,
        transform=axis_for_spectrogram.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"),
    )
plt.setp(observed_spectrogram_axis.get_xticklabels(), visible=False)
model_spectrogram_axis.set_xlabel("Sweep index within chunk")
figure.colorbar(
    mesh, ax=[observed_spectrogram_axis, model_spectrogram_axis],
    label="Coincidence rate [Hz]")

if arguments.output_path is not None:
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output_path, bbox_inches="tight")
    print(f"Saved {arguments.output_path}")
else:
    plt.show()
