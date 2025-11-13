from datetime import datetime
from typing import Optional

import numpy as np
import spiceypy

from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData, RectangularCoords, \
    InputRectangularPointingSet, GlowsL3eRectangularMapInputData, RectangularSpectralIndexMapData, SpectralIndexMapData


def create_rectangular_intensity_map_data(epoch=None, epoch_delta=None, lon=None, lat=None, energy=None,
                                          energy_delta=None, flux=None,
                                          intensity_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    epoch_delta = epoch_delta if epoch_delta is not None else np.ma.array([86400 * 1e9])
    flux = flux if flux is not None else np.full((len(epoch), len(energy), len(lon), len(lat)), fill_value=1)
    intensity_stat_unc = intensity_stat_unc if intensity_stat_unc is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)

    if isinstance(flux, np.ndarray):

        more_real_flux = flux

    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return RectangularIntensityMapData(
        intensity_map_data=IntensityMapData(
            epoch=epoch,
            epoch_delta=epoch_delta,
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(flux, 0),
            obs_date=np.ma.array(np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))),
            obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_intensity=flux,
            ena_intensity_stat_unc=intensity_stat_unc,
            ena_intensity_sys_err=np.full_like(flux, 0)),
        coords=RectangularCoords(
            latitude_delta=np.full_like(lat, 0),
            latitude_label=lat.astype(str),
            longitude_delta=np.full_like(lon, 0),
            longitude_label=lon.astype(str),

        )

    )


def create_rectangular_spectral_index_map_data(epoch=None, epoch_delta=None, lon=None, lat=None, energy=None,
                                               energy_delta=None, spectral_index=None,
                                               spectral_index_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
    epoch_delta = epoch_delta if epoch_delta is not None else np.ma.array([86400 * 1e9])
    spectral_index = spectral_index if spectral_index is not None else np.full((len(epoch), len(lon), len(lat)),
                                                                               fill_value=1)
    spectral_index_stat_unc = spectral_index_stat_unc if spectral_index_stat_unc is not None else np.full(
        (len(epoch), len(energy), len(lon), len(lat)),
        fill_value=1)

    if isinstance(spectral_index, np.ndarray):
        more_real_flux = spectral_index
    else:
        more_real_flux = np.full((len(epoch), 9, len(lon), len(lat)), fill_value=1)

    return RectangularSpectralIndexMapData(
        spectral_index_map_data=SpectralIndexMapData(
            epoch=epoch,
            epoch_delta=epoch_delta,
            energy=energy,
            energy_delta_plus=energy_delta,
            energy_delta_minus=energy_delta,
            energy_label=np.array(["energy"]),
            latitude=lat,
            longitude=lon,
            exposure_factor=np.full_like(spectral_index, 0),
            obs_date=np.ma.array(np.full(more_real_flux.shape, datetime(year=2010, month=1, day=1))),
            obs_date_range=np.ma.array(np.full_like(more_real_flux, 0)),
            solid_angle=np.full_like(more_real_flux, 0),
            ena_spectral_index=spectral_index,
            ena_spectral_index_stat_unc=spectral_index_stat_unc
        ),
        coords=RectangularCoords(
            latitude_delta=np.full_like(lat, 0),
            latitude_label=lat.astype(str),
            longitude_delta=np.full_like(lon, 0),
            longitude_label=lon.astype(str),
        )
    )


def create_l1c_pset(
    epoch: datetime = datetime(2025, 4, 15, 12),
    repointing: int = 1,
    energy_steps: np.ndarray = np.array([1]),
    exposures: Optional[np.ndarray] = None,
) -> InputRectangularPointingSet:
    epoch_j2000 = np.array([spiceypy.datetime2et(epoch)]) * 1e9
    exposures = exposures or np.full(shape=(1, energy_steps.shape[0], 3600), fill_value=1.)
    return InputRectangularPointingSet(epoch, epoch_j2000, repointing, exposures, energy_steps)


def create_l3e_pset(
    epoch: datetime = datetime(2025, 4, 15, 12),
    repointing: int = 1,
    energy_steps = np.array([0.5, 5.0, 12.0]),
    spin_angle = np.arange(0, 360),
    sp: Optional[np.array] = None,
) -> GlowsL3eRectangularMapInputData:
    epoch_j2000 = np.array([spiceypy.datetime2et(epoch)]) * 1e9
    sp = sp or np.full(shape=(1, len(energy_steps), len(spin_angle)), fill_value=0.5)
    return GlowsL3eRectangularMapInputData(epoch, epoch_j2000, repointing, energy_steps, spin_angle, sp)
