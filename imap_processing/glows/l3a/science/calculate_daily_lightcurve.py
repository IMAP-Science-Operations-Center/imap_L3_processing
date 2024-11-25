import numpy as np
from numpy import ndarray

from imap_processing.glows.l3a.science.bad_angle_flag_configuration import BadAngleFlagConfiguration
from imap_processing.glows.l3a.science.time_independent_background_lookup_table import \
    TimeIndependentBackgroundLookupTable


def rebin_lightcurve(time_independent_background_table: TimeIndependentBackgroundLookupTable,
                     bad_angle_flag_configuration: BadAngleFlagConfiguration,
                     photon_flux: np.ndarray,
                     latitudes: np.ndarray,
                     longitudes: np.ndarray,
                     flags: np.ndarray,
                     exposure_times: np.ndarray, output_size: int, background: ndarray[float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    photon_flux_minus_time_independent_background = photon_flux - time_independent_background_table.lookup(
        lat=latitudes, lon=longitudes)

    masked = bad_angle_flag_configuration.evaluate_flags(flags)
    included = np.logical_not(masked)
    filtered_exposure_times = exposure_times * included
    exposure_time_by_bins = np.reshape(filtered_exposure_times, (output_size, -1))

    exposure_times_rebinned = np.sum(exposure_time_by_bins, axis=-1)

    filtered_photon_flux = photon_flux_minus_time_independent_background * filtered_exposure_times
    split_by_bins = np.reshape(filtered_photon_flux, (output_size, -1))

    rebinned_photon_flux = np.full(shape=(output_size,), fill_value=np.nan, dtype=filtered_photon_flux.dtype)

    np.divide(np.sum(split_by_bins, axis=-1), exposure_times_rebinned, out=rebinned_photon_flux,
              where=exposure_times_rebinned != 0)

    rebinned_photon_flux = rebinned_photon_flux - background
    return rebinned_photon_flux, exposure_times_rebinned


def calculate_spin_angles(number_of_bins: int, spin_angles: np.ndarray) -> np.ndarray:
    binned = spin_angles.reshape((number_of_bins, -1))
    unwrapped = np.unwrap(binned, axis=-1, period=360)
    return np.mod(np.mean(unwrapped, axis=-1), 360)
