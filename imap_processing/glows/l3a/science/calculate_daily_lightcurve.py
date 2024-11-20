import numpy as np
from numpy import ndarray


def rebin_lightcurve(photon_flux: np.ndarray, flags: np.ndarray,
                     exposure_times: np.ndarray, output_size: int, background: ndarray[float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    included = np.all(flags == False, axis=-2)

    filtered_exposure_times = exposure_times * included
    exposure_time_by_bins = np.reshape(filtered_exposure_times, (output_size, -1))

    exposure_times_rebinned = np.sum(exposure_time_by_bins, axis=-2)

    filtered_photon_flux = photon_flux * exposure_times
    split_by_bins = np.reshape(filtered_photon_flux, (output_size, -1))

    photon_flux_rebinned = np.sum(split_by_bins, axis=-2) / exposure_times_rebinned

    photon_flux_rebinned = photon_flux_rebinned - background
    return photon_flux_rebinned, exposure_times_rebinned
