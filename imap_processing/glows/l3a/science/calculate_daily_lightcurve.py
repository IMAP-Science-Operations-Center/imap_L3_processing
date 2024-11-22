import numpy as np
from numpy import ndarray


def rebin_lightcurve(photon_flux: np.ndarray, flags: np.ndarray,
                     exposure_times: np.ndarray, output_size: int, background: ndarray[float]
                     ) -> tuple[np.ndarray, np.ndarray]:
    included = np.all(flags == False, axis=-2)

    filtered_exposure_times = exposure_times * included
    exposure_time_by_bins = np.reshape(filtered_exposure_times, (output_size, -1))

    exposure_times_rebinned = np.sum(exposure_time_by_bins, axis=-1)

    filtered_photon_flux = photon_flux * filtered_exposure_times
    split_by_bins = np.reshape(filtered_photon_flux, (output_size, -1))

    rebinned_photon_flux = np.full(shape=(output_size,), fill_value=np.nan, dtype=filtered_photon_flux.dtype)

    np.divide(np.sum(split_by_bins, axis=-1), exposure_times_rebinned, out=rebinned_photon_flux,
              where=exposure_times_rebinned != 0)

    rebinned_photon_flux = rebinned_photon_flux - background
    return rebinned_photon_flux, exposure_times_rebinned
