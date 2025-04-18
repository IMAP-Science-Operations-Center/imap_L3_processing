from dataclasses import dataclass

from imap_data_access import upload

from imap_l3_processing.processor import Processor
from imap_l3_processing.ultra.l3.models import UltraL3SurvivalCorrectedDataProduct, UltraGlowsL3eData, UltraL1CPSet
from imap_l3_processing.ultra.l3.science.ultra_survival_probability import Sensor, UltraSurvivalProbability, \
    UltraSurvivalProbabilitySkyMap
from imap_l3_processing.ultra.l3.ultra_l3_dependencies import UltraL3Dependencies
from imap_l3_processing.utils import save_data


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
        # spice furnish?
        combined_psets = combine_glows_l3e_ultra_l1c(deps.ultra_l1c_pset, deps.glows_l3e_sp)
        survival_probability_psets = [UltraSurvivalProbability(*_dep) for _dep in combined_psets]

        map_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        corrected_skymap = UltraSurvivalProbabilitySkyMap(survival_probability_psets)
        # accumulate over combined psets and call UltraSurvivalProbability

        # use psets to build sp skymap
        # divide ultra l2 flux map by sp skymap

        # build dataproduct


def combine_glows_l3e_ultra_l1c(ultra_l1c: list[UltraL1CPSet], glows_l3e: list[UltraGlowsL3eData]):
    raise NotImplementedError


@dataclass
class UltraMapDescriptorParts:
    sensor: Sensor
    grid_size: int


def parse_map_descriptor(descriptor: str) -> UltraMapDescriptorParts:
    sensor = Sensor(descriptor[:2])
    grid_size = int(descriptor.split("-")[4][0])

    return UltraMapDescriptorParts(sensor, grid_size)
