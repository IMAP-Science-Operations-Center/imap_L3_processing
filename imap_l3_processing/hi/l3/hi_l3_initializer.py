import dataclasses
import logging
from pathlib import Path

import imap_data_access

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, Sensor, SurvivalCorrection, SpinPhase
from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.utils import SpiceKernelTypes, furnish_spice_metakernel

logger = logging.getLogger(__name__)

HI_SP_SPICE_KERNELS: list[SpiceKernelTypes] = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
]

combined_descriptors = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
]

spectral_index_descriptors = [
    "h45-spx-h-hf-sp-ram-hae-6deg-1yr",
    "h45-spx-h-hf-sp-anti-hae-6deg-1yr",
    "h45-spx-h-hf-sp-full-hae-6deg-6mo",
    "h90-spx-h-hf-sp-ram-hae-6deg-1yr",
    "h90-spx-h-hf-sp-anti-hae-6deg-1yr",
    "h90-spx-h-hf-sp-full-hae-6deg-6mo",
    "h45-spx-h-hf-sp-ram-hae-4deg-1yr",
    "h45-spx-h-hf-sp-anti-hae-4deg-1yr",
    "h45-spx-h-hf-sp-full-hae-4deg-6mo",
    "h90-spx-h-hf-sp-ram-hae-4deg-1yr",
    "h90-spx-h-hf-sp-anti-hae-4deg-1yr",
    "h90-spx-h-hf-sp-full-hae-4deg-6mo",
    "hic-spx-h-hf-sp-full-hae-4deg-1yr",
    "hic-spx-h-hf-sp-full-hae-6deg-1yr",
]

HI_SP_MAP_DESCRIPTORS_WITH_NON_SCIENTIFIC_VALUES = [
    "h45-ena-h-sf-sp-full-hae-6deg-6mo",
    "h45-ena-h-sf-sp-full-hae-4deg-6mo",
]

HI_SP_MAP_DESCRIPTORS = [
    "h90-ena-h-sf-sp-ram-hae-6deg-1yr",
    "h90-ena-h-hf-sp-ram-hae-6deg-1yr",
    "h90-ena-h-sf-sp-anti-hae-6deg-1yr",
    "h90-ena-h-hf-sp-anti-hae-6deg-1yr",

    "h90-ena-h-sf-sp-full-hae-6deg-6mo",
    "h90-ena-h-hf-sp-full-hae-6deg-6mo",

    "h45-ena-h-sf-sp-ram-hae-6deg-1yr",
    "h45-ena-h-hf-sp-ram-hae-6deg-1yr",
    "h45-ena-h-sf-sp-anti-hae-6deg-1yr",
    "h45-ena-h-hf-sp-anti-hae-6deg-1yr",

    "h45-ena-h-hf-sp-full-hae-6deg-6mo",

    "h90-ena-h-sf-sp-ram-hae-4deg-1yr",
    "h90-ena-h-hf-sp-ram-hae-4deg-1yr",
    "h90-ena-h-sf-sp-anti-hae-4deg-1yr",
    "h90-ena-h-hf-sp-anti-hae-4deg-1yr",

    "h90-ena-h-sf-sp-full-hae-4deg-6mo",
    "h90-ena-h-hf-sp-full-hae-4deg-6mo",

    "h45-ena-h-sf-sp-ram-hae-4deg-1yr",
    "h45-ena-h-hf-sp-ram-hae-4deg-1yr",
    "h45-ena-h-sf-sp-anti-hae-4deg-1yr",
    "h45-ena-h-hf-sp-anti-hae-4deg-1yr",

    "h45-ena-h-hf-sp-full-hae-4deg-6mo",
]


class HiL3Initializer(MapInitializer):
    def __init__(self):
        sp_hi45_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor=f'survival-probability-hi-45',
            version='latest'
        )
        self.glows_hi45_file_by_repoint = {int(r["repointing"]): Path(r["file_path"]).name for r in sp_hi45_query_result}

        sp_hi90_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor=f'survival-probability-hi-90',
            version='latest'
        )
        self.glows_hi90_file_by_repoint = {int(r["repointing"]): Path(r["file_path"]).name for r in sp_hi90_query_result}

        hi_l2_query_result = imap_data_access.query(instrument='hi', data_level='l2')
        hi_l3_query_result = imap_data_access.query(instrument='hi', data_level='l3')
        super().__init__("hi", hi_l2_query_result, hi_l3_query_result)

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(start_date=map_to_produce.input_metadata.start_date,
                                 end_date=map_to_produce.input_metadata.end_date, kernel_types=HI_SP_SPICE_KERNELS)

    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        match descriptor:
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    spin_phase=SpinPhase.RamOnly | SpinPhase.AntiRamOnly):
                return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected)]
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    spin_phase=SpinPhase.FullSpin):
                return [
                    dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                        spin_phase=SpinPhase.AntiRamOnly),
                    dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                        spin_phase=SpinPhase.RamOnly),
                ]
            case _:
                raise ValueError("Expected map to be produced to use a single sensor!")

    def _collect_glows_psets_by_repoint(self, descriptor: MapDescriptorParts) -> dict[int, str]:
        match descriptor:
            case MapDescriptorParts(sensor=Sensor.Hi45):
                return self.glows_hi45_file_by_repoint
            case MapDescriptorParts(sensor=Sensor.Hi90):
                return self.glows_hi90_file_by_repoint
            case _:
                raise ValueError("Expected map to be produced to use a single sensor!")
