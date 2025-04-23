from dataclasses import dataclass

from imap_data_access import upload
from imap_processing.spice import geometry

from imap_l3_processing.processor import Processor
from imap_l3_processing.ultra.l3.models import UltraL3SurvivalCorrectedDataProduct
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

    def _process_survival_probability(self, deps: UltraL3Dependencies) -> UltraL3SurvivalCorrectedDataProduct:
        combined_psets = combine_glows_l3e_with_l1c_pointing(deps.glows_l3e_sp, deps.ultra_l1c_pset, )
        survival_probability_psets = [UltraSurvivalProbability(_l1c, _l3e) for _l1c, _l3e in
                                      combined_psets]

        map_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        corrected_skymap = UltraSurvivalProbabilitySkyMap(survival_probability_psets, geometry.SpiceFrame.ECLIPJ2000, )
        survival_probability_map = corrected_skymap.to_dataset()["exposure_weighted_survival_probabilities"].values
        input_data = deps.ultra_l2_map

        corrected_intensity = input_data.ena_intensity / survival_probability_map
        corrected_stat_unc = input_data.ena_intensity_stat_unc / survival_probability_map
        corrected_sys_unc = input_data.ena_intensity_sys_err / survival_probability_map

        return UltraL3SurvivalCorrectedDataProduct(
            input_metadata=self.input_metadata.to_upstream_data_dependency(self.input_metadata.descriptor),
            ena_intensity_stat_unc=corrected_stat_unc,
            ena_intensity_sys_err=corrected_sys_unc,
            ena_intensity=corrected_intensity,
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
            pixel_index=input_data.pixel_index,
            pixel_index_label=input_data.pixel_index_label
        )


@dataclass
class UltraMapDescriptorParts:
    grid_size: int


def parse_map_descriptor(descriptor: str) -> UltraMapDescriptorParts:
    grid_size = int(descriptor.split("-")[4][0])

    return UltraMapDescriptorParts(grid_size)
