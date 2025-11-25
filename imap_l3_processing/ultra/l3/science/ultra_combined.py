from imap_l3_processing.maps.map_models import HealPixIntensityMapData


def combine_sensors(u45: HealPixIntensityMapData, u90: HealPixIntensityMapData) -> HealPixIntensityMapData:
    intensity = (((u45.intensity_map_data.ena_intensity * u45.intensity_map_data.exposure_factor) +
                 (u90.intensity_map_data.ena_intensity * u90.intensity_map_data.exposure_factor)) /
                 (u45.intensity_map_data.exposure_factor + u90.intensity_map_data.exposure_factor))

    intensity_systematic_error = (((u45.intensity_map_data.ena_intensity_sys_err * u45.intensity_map_data.exposure_factor) +
                 (u90.intensity_map_data.ena_intensity_sys_err * u90.intensity_map_data.exposure_factor)) /
                 (u45.intensity_map_data.exposure_factor + u90.intensity_map_data.exposure_factor))

    intensity_static_error = (
                ((u45.intensity_map_data.ena_intensity_stat_uncert * u45.intensity_map_data.exposure_factor) +
                 (u90.intensity_map_data.ena_intensity_stat_uncert * u90.intensity_map_data.exposure_factor)) /
                 (u45.intensity_map_data.exposure_factor + u90.intensity_map_data.exposure_factor))