from datetime import datetime

import numpy as np
import spiceypy

from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData, RectangularCoords, \
    InputRectangularPointingSet, GlowsL3eRectangularMapInputData


def create_h1_l3_data(epoch=None, lon=None, lat=None, energy=None, energy_delta=None, flux=None,
                      intensity_stat_unc=None):
    lon = lon if lon is not None else np.array([1.0])
    lat = lat if lat is not None else np.array([1.0])
    energy = energy if energy is not None else np.array([1.0])
    energy_delta = energy_delta if energy_delta is not None else np.full((len(energy), 2), 1)
    epoch = epoch if epoch is not None else np.ma.array([datetime.now()])
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
            epoch_delta=np.ma.array([0]),
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


def create_l1c_pset(epoch: datetime) -> InputRectangularPointingSet:
    epoch_j2000 = np.array([spiceypy.datetime2et(epoch)]) * 1e9
    energy_steps = np.array([1])
    exposures = np.full(shape=(1, energy_steps.shape[0], 3600), fill_value=1.)
    return InputRectangularPointingSet(epoch, epoch_j2000, exposures, energy_steps)


def create_l3e_pset(epoch) -> GlowsL3eRectangularMapInputData:
    energy_steps = np.array([0.5, 5.0, 12.0])
    spin_angle = np.arange(0, 360)
    sp = np.full(shape=(1, len(energy_steps), len(spin_angle)), fill_value=0.5)
    return GlowsL3eRectangularMapInputData(epoch, energy_steps, spin_angle, sp)
