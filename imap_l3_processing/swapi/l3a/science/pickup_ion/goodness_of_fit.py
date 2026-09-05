import numpy as np
from numpy.typing import NDArray

from imap_l3_processing.swapi.constants import SWAPI_BACKGROUND_RATE

MAX_CUTOFF_SPEED_KMS = 550.0
MAX_MEAN_RELATIVE_ERROR = 0.12
MAX_PAST_PEAK_RATIO = 0.4
CUTOFF_DROP_RATIO = 0.25


def is_good_fit(
    esa_energies: NDArray,
    model_rates: NDArray,
    observed_rates: NDArray,
    cutoff_speed_kms: float,
    min_fitting_energy: float,
) -> bool:
    """
    Evaluate whether a PUI fit is a good fit.

    For the goodness of fit criteria, see [docs/swapi/pickup-ion.md].

    Parameters
    ----------
    esa_energies : ndarray of floats
        The 62 coarse-step ESA energies.
    model_rates : ndarray of floats
        The (50, 62) array of model count rates, *excluding* the background rate.
    observed_rates : ndarray of floats
        The (50, 62) array of observed count rates.
    cutoff_speed_kms : scalar float
        The model cutoff speed in km/s.
    min_fitting_energy: scalar float
        The minimum energy required to fit the model.
    """
    if esa_energies.shape != (62,):
         raise ValueError(esa_energies.shape)

    if model_rates.shape != (50, 62,):
         raise ValueError(model_rates.shape)

    if observed_rates.shape != (50, 62,):
         raise ValueError(observed_rates.shape)

    chunk_mean_model_rates = model_rates.mean(axis=0)
    chunk_mean_observed_rates = observed_rates.mean(axis=0)

    peak_model_rate = chunk_mean_model_rates.max()

    upper_energy_limit = esa_energies[
        chunk_mean_model_rates >= peak_model_rate * CUTOFF_DROP_RATIO
    ].max()
    past_range = esa_energies > upper_energy_limit
    in_range = ~past_range & (esa_energies > min_fitting_energy)

    mean_absolute_percent_error = (
        np.abs(chunk_mean_model_rates + SWAPI_BACKGROUND_RATE - chunk_mean_observed_rates)
        / (chunk_mean_model_rates + SWAPI_BACKGROUND_RATE)
    )[in_range].mean()

    past_cutoff_ratio = (
        chunk_mean_observed_rates[past_range].mean()
        / chunk_mean_model_rates.max()
    )

    return (
        (cutoff_speed_kms < MAX_CUTOFF_SPEED_KMS)
        and (mean_absolute_percent_error < MAX_MEAN_RELATIVE_ERROR)
        and (past_cutoff_ratio < MAX_PAST_PEAK_RATIO)
    )