import dataclasses
from pathlib import Path

import imap_data_access

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, SurvivalCorrection
from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.utils import SpiceKernelTypes, furnish_spice_metakernel

LO_SP_MAP_KERNELS = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
]

LO_SP_MAP_DESCRIPTORS = [
    "l090-ena-h-sf-sp-ram-hae-6deg-1yr",
    "l090-ena-h-hf-sp-ram-hae-6deg-1yr",
]

LO_ISN_MAP_DESCRIPTORS = [
    "l090-isn-h-sf-nsp-ram-hae-6deg-1yr",
    "l090-isn-o-sf-nsp-ram-hae-6deg-1yr"
]

LO_SPECTRAL_INDEX_MAP_DESCRIPTORS = [
    "l090-spx-h-sf-nsp-ram-hae-6deg-1yr",
    "l090-spx-h-sf-sp-ram-hae-6deg-1yr",
    "l090-spx-h-hf-nsp-ram-hae-6deg-1yr",
    "l090-spx-h-hf-sp-ram-hae-6deg-1yr",
    "l090-spx-h-hk-nsp-ram-hae-6deg-1yr",
    "l090-spx-h-hk-sp-ram-hae-6deg-1yr",
    "l090-spxnbs-h-sf-nsp-ram-hae-6deg-1yr",
    "l090-spxnbs-h-hk-nsp-ram-hae-6deg-1yr",
    "lxxx-spxnbs-h-sf-sp-ram-hae-6deg-1yr",
    "lxxx-spxnbs-h-hk-nsp-ram-hae-6deg-1yr",
]


class LoInitializer(MapInitializer):
    def __init__(self):
        glows_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor="survival-probability-lo",
            version="latest"
        )
        self.glows_files_by_repointing = {int(r["repointing"]): Path(r["file_path"]).name for r in glows_query_result}

        lo_l2_query_result = imap_data_access.query(instrument="lo", data_level="l2")
        lo_l3_query_result = imap_data_access.query(instrument="lo", data_level="l3")

        super().__init__("lo", lo_l2_query_result, lo_l3_query_result)

    def _collect_glows_psets_by_repoint(self, descriptor: str) -> dict[int, str]:
        return self.glows_files_by_repointing

    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected)]

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(start_date=map_to_produce.input_metadata.start_date,
                                 end_date=map_to_produce.input_metadata.end_date, kernel_types=LO_SP_MAP_KERNELS)
