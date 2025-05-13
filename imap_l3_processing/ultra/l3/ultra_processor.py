from dataclasses import dataclass

from imap_data_access import upload
from imap_processing.spice import geometry

from imap_l3_processing.maps.map_models import HealPixIntensityDataProduct, HealPixIntensityMapData, IntensityMapData, \
    HealPixCoords
from imap_l3_processing.processor import Processor
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import UltraSurvivalProbabilitySkyMap, \
    UltraSurvivalProbability
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies
from imap_l3_processing.utils import save_data, combine_glows_l3e_with_l1c_pointing


class UltraProcessor(Processor):
    def process(self):
        if "survival" in self.input_metadata.descriptor:
            deps = UltraL3Dependencies.fetch_dependencies(self.dependencies)
            data_product = self._process_survival_probability(deps)
            data_product_path = save_data(data_product)
            upload(data_product_path)
        else:
            raise NotImplementedError

    def _process_survival_probability(self, deps: UltraL3Dependencies) -> HealPixIntensityDataProduct:
        combined_psets = combine_glows_l3e_with_l1c_pointing(deps.glows_l3e_sp, deps.ultra_l1c_pset, )
        survival_probability_psets = [UltraSurvivalProbability(_l1c, _l3e) for _l1c, _l3e in
                                      combined_psets]

        intensity_data = deps.ultra_l2_map.intensity_map_data
        coords = deps.ultra_l2_map.coords
        corrected_skymap = UltraSurvivalProbabilitySkyMap(survival_probability_psets, geometry.SpiceFrame.ECLIPJ2000,
                                                          coords.nside)
        survival_probability_map = corrected_skymap.to_dataset()["exposure_weighted_survival_probabilities"].values

        corrected_intensity = intensity_data.ena_intensity / survival_probability_map
        corrected_stat_unc = intensity_data.ena_intensity_stat_unc / survival_probability_map
        corrected_sys_unc = intensity_data.ena_intensity_sys_err / survival_probability_map

        return HealPixIntensityDataProduct(
            input_metadata=self.input_metadata.to_upstream_data_dependency(self.input_metadata.descriptor),
            data=HealPixIntensityMapData(
                intensity_map_data=IntensityMapData(
                    ena_intensity_stat_unc=corrected_stat_unc,
                    ena_intensity_sys_err=corrected_sys_unc,
                    ena_intensity=corrected_intensity,
                    epoch=intensity_data.epoch,
                    epoch_delta=intensity_data.epoch_delta,
                    energy=intensity_data.energy,
                    energy_delta_plus=intensity_data.energy_delta_plus,
                    energy_delta_minus=intensity_data.energy_delta_minus,
                    energy_label=intensity_data.energy_label,
                    latitude=intensity_data.latitude,
                    longitude=intensity_data.longitude,
                    exposure_factor=intensity_data.exposure_factor,
                    obs_date=intensity_data.obs_date,
                    obs_date_range=intensity_data.obs_date_range,
                    solid_angle=intensity_data.solid_angle,
                ),
                coords=HealPixCoords(
                    pixel_index=coords.pixel_index,
                    pixel_index_label=coords.pixel_index_label,
                ),
            )
        )


@dataclass
class UltraMapDescriptorParts:
    grid_size: int


def parse_map_descriptor(descriptor: str) -> UltraMapDescriptorParts:
    grid_size = int(descriptor.split("-")[4][0])

    return UltraMapDescriptorParts(grid_size)
