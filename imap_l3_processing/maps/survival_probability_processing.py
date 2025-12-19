import imap_data_access
import numpy as np
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import ReferenceFrame, map_descriptor_parts_to_string
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

    cg_corrected = l2_descriptor_parts.reference_frame == ReferenceFrame.Heliospheric
    for l1c, glows_l3e in combined_glows:
        pointing_sets.append(RectangularSurvivalProbabilityPointingSet(
            l1c, l2_descriptor_parts.sensor, l2_descriptor_parts.spin_phase, glows_l3e,
            survival_probabilities_dependencies.l2_data.intensity_map_data.energy, cg_corrected=cg_corrected))
    assert len(pointing_sets) > 0

    survival_sky_map = RectangularSurvivalProbabilitySkyMap(pointing_sets, int(l2_descriptor_parts.grid),
                                                            spice_frame_name)

    survival_dataset = survival_sky_map.to_dataset()

    input_data = survival_probabilities_dependencies.l2_data.intensity_map_data
    survival_probabilities = survival_dataset["exposure_weighted_survival_probabilities"].values

    time = input_data.epoch[0].strftime("%Y%m%d")

    input_l2_descriptor = map_descriptor_parts_to_string(survival_probabilities_dependencies.l2_map_descriptor_parts)
    output_csv_dir = imap_data_access.config["DATA_DIR"] / "sp_maps" / f"sp_map_{input_l2_descriptor}_{time}"
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    for i, energy in enumerate(input_data.energy):
        output_csv_path = output_csv_dir / f"sp_map_{energy}_{input_l2_descriptor}.csv"
        np.savetxt(output_csv_path, survival_probabilities[0, i, ...], delimiter=",")

    survival_corrected_intensity = input_data.ena_intensity / survival_probabilities
    corrected_stat_uncert = input_data.ena_intensity_stat_uncert / survival_probabilities
    corrected_sys_err = input_data.ena_intensity_sys_err / survival_probabilities

    map_data = RectangularIntensityMapData(
        intensity_map_data=IntensityMapData(
            ena_intensity_stat_uncert=corrected_stat_uncert,
            ena_intensity_sys_err=corrected_sys_err,
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

    if input_data.bg_intensity is not None:
        map_data.intensity_map_data.bg_intensity = input_data.bg_intensity / survival_probabilities
        map_data.intensity_map_data.bg_intensity_sys_err = input_data.bg_intensity_sys_err / survival_probabilities
        map_data.intensity_map_data.bg_intensity_stat_uncert = input_data.bg_intensity_stat_uncert / survival_probabilities

    return map_data
