import numpy as np


def rebin_lightcurve(photon_flux: np.ndarray, flags: np.ndarray,
                     exposure_times: np.ndarray, output_size: int
                     )->tuple[np.ndarray, np.ndarray]:
    flagged = flags.any(axis=0)
    filtered_photon_flux = np.ma.array(photon_flux, mask=flagged)
    split_by_bins = np.reshape(filtered_photon_flux, (output_size, -1))

    exposure_times = np.ma.array(exposure_times, mask=flagged)
    exposure_time_by_bins = np.reshape(exposure_times, (output_size, -1))

    photon_flux_rebinned, exposure_times_rebinned = np.ma.average(split_by_bins, weights=exposure_time_by_bins, returned=True, axis=1)
    return photon_flux_rebinned.filled(0), exposure_times_rebinned.filled(0)
