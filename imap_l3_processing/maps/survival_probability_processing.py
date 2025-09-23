from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import Sensor
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, IntensityMapData
from imap_l3_processing.maps.rectangular_survival_probability import RectangularSurvivalProbabilityPointingSet, \
    RectangularSurvivalProbabilitySkyMap
from imap_l3_processing.utils import combine_glows_l3e_with_l1c_pointing


def process_survival_probabilities(survival_probabilities_dependencies: HiLoL3SurvivalDependencies,
                                   spice_frame_name: SpiceFrame) \
        -> RectangularIntensityMapData:
    l2_descriptor_parts = survival_probabilities_dependencies.l2_map_descriptor_parts

    combined_glows = combine_glows_l3e_with_l1c_pointing(survival_probabilities_dependencies.glows_l3e_data,
                                                         survival_probabilities_dependencies.l1c_data)
    pointing_sets = []
    input_data = survival_probabilities_dependencies.l2_data.intensity_map_data

    for l1c, glows_l3e in combined_glows:
        if (survival_probabilities_dependencies.l2_map_descriptor_parts.sensor == Sensor.Lo or
                survival_probabilities_dependencies.l2_map_descriptor_parts.sensor == Sensor.Lo90):
            l1c.exposure_times = l1c.exposure_times.sum(axis=3)

        pointing_sets.append(RectangularSurvivalProbabilityPointingSet(
            l1c, l2_descriptor_parts.sensor, l2_descriptor_parts.spin_phase, glows_l3e,
            input_data.energy))
    assert len(pointing_sets) > 0

    survival_sky_map = RectangularSurvivalProbabilitySkyMap(pointing_sets, int(l2_descriptor_parts.grid),
                                                            spice_frame_name)

    survival_dataset = survival_sky_map.to_dataset()

    input_data = survival_probabilities_dependencies.l2_data.intensity_map_data
    survival_probabilities = survival_dataset["exposure_weighted_survival_probabilities"].values

    survival_corrected_intensity = input_data.ena_intensity / survival_probabilities
    corrected_stat_unc = input_data.ena_intensity_stat_unc / survival_probabilities
    corrected_sys_unc = input_data.ena_intensity_sys_err / survival_probabilities

    return RectangularIntensityMapData(
        intensity_map_data=IntensityMapData(
            ena_intensity_stat_unc=corrected_stat_unc,
            ena_intensity_sys_err=corrected_sys_unc,
            ena_intensity=survival_corrected_intensity,
            epoch=input_data.epoch,
            epoch_delta=input_data.epoch_delta,
            energy=input_data.energy,
            energy_delta_plus=input_data.energy_delta_plus,
            energy_delta_minus=input_data.energy_delta_minus,
            energy_label=input_data.energy_label,
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            exposure_factor=input_data.exposure_factor,
            obs_date=input_data.obs_date,
            obs_date_range=input_data.obs_date_range,
            solid_angle=input_data.solid_angle,
        ),
        coords=survival_probabilities_dependencies.l2_data.coords
    )
