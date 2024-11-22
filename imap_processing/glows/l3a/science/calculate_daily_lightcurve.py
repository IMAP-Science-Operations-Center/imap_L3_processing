import numpy as np
from numpy import ndarray

from imap_processing.glows.l3a.science.time_independent_background_lookup_table import \
    TimeIndependentBackgroundLookupTable


def rebin_lightcurve(time_independent_background_table: TimeIndependentBackgroundLookupTable,
                     photon_flux: np.ndarray,
                     latitudes: np.ndarray,
                     longitudes: np.ndarray,
                     flags: np.ndarray,
                     exposure_times: np.ndarray, output_size: int, background: ndarray[float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    photon_flux_minus_time_independent_background = photon_flux - time_independent_background_table.lookup(
        lat=latitudes, lon=longitudes)

    included = np.all(flags == False, axis=-2)
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
