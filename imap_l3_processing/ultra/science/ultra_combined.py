from imap_l3_processing.maps.map_models import HealPixIntensityMapData, RectangularIntensityMapData, IntensityMapData, \
    HealPixCoords

def map_combined_rectangular_quantities_to_healpix_intensity_map(healpix_map: HealPixIntensityMapData,
                                                                 rectangular_map: RectangularIntensityMapData) -> HealPixIntensityMapData:
    healpix_map_intensities = healpix_map.intensity_map_data
    rectangular_intensity_data = rectangular_map.intensity_map_data

    healpix_map_data = HealPixIntensityMapData(
        intensity_map_data=IntensityMapData(
            ena_intensity_stat_uncert=healpix_map_intensities.ena_intensity_stat_uncert,
            ena_intensity_sys_err=healpix_map_intensities.ena_intensity_sys_err,
            ena_intensity=healpix_map_intensities.ena_intensity,
            epoch=rectangular_intensity_data.epoch,
            epoch_delta=rectangular_intensity_data.epoch_delta,
            energy=rectangular_intensity_data.energy,
            energy_delta_plus=rectangular_intensity_data.energy_delta_plus,
            energy_delta_minus=rectangular_intensity_data.energy_delta_minus,
            energy_label=rectangular_intensity_data.energy_label,
            latitude=rectangular_intensity_data.latitude,
            longitude=rectangular_intensity_data.longitude,
            exposure_factor=rectangular_intensity_data.exposure_factor,
            obs_date=rectangular_intensity_data.obs_date,
            obs_date_range=rectangular_intensity_data.obs_date_range,
            solid_angle=rectangular_intensity_data.solid_angle,
        ),
        coords=HealPixCoords(
            pixel_index=healpix_map.coords.pixel_index,
            pixel_index_label=healpix_map.coords.pixel_index_label,
        )
    )
    return healpix_map_data