import numpy as np

from imap_l3_processing.maps.map_models import calculate_datetime_weighted_average, \
    SpectralIndexMapData, SpectralIndexDependencies
from imap_l3_processing.maps.spectral_fit import spectral_fit


def process_spectral_index(spectral_index_dependencies: SpectralIndexDependencies) \
        -> SpectralIndexMapData:
    input_data = spectral_index_dependencies.map_data
    intensity_data = input_data.intensity_map_data

    energy = intensity_data.energy
    fluxes = intensity_data.ena_intensity
    variances = np.square(intensity_data.ena_intensity_stat_unc)

    gammas, errors = spectral_fit(fluxes, variances, energy)

    min_energy = intensity_data.energy[0] - intensity_data.energy_delta_minus[0]
    max_energy = intensity_data.energy[-1] + intensity_data.energy_delta_plus[-1]
    mean_energy = np.sqrt(min_energy * max_energy)

    new_energy_label = f"{min_energy} - {max_energy} keV"

    mean_obs_date = calculate_datetime_weighted_average(intensity_data.obs_date,
                                                        weights=intensity_data.exposure_factor,
                                                        axis=1, keepdims=True)
    mean_obs_date_range = np.ma.average(intensity_data.obs_date_range, weights=intensity_data.exposure_factor,
                                        axis=1,
                                        keepdims=True)
    total_exposure_factor = np.sum(intensity_data.exposure_factor, axis=1, keepdims=True)

    return SpectralIndexMapData(
        ena_spectral_index_stat_unc=errors,
        ena_spectral_index=gammas,
        epoch=intensity_data.epoch,
        epoch_delta=intensity_data.epoch_delta,
        energy=np.array([mean_energy]),
        energy_delta_plus=np.array([max_energy - mean_energy]),
        energy_delta_minus=np.array([mean_energy - min_energy]),
        energy_label=np.array([new_energy_label]),
        latitude=intensity_data.latitude,
        longitude=intensity_data.longitude,
        exposure_factor=total_exposure_factor,
        obs_date=mean_obs_date,
        obs_date_range=mean_obs_date_range,
        solid_angle=intensity_data.solid_angle,
    )
