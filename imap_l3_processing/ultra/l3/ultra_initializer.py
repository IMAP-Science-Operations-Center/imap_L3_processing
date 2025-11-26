import dataclasses
from pathlib import Path

import imap_data_access

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, SurvivalCorrection
from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.utils import furnish_spice_metakernel, SpiceKernelTypes

ULTRA_SP_SPICE_KERNELS = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
]

ULTRA_SP_MAP_DESCRIPTORS = [
    "ulc-ena-h-hf-sp-full-hae-2deg-3mo",
    "ulc-ena-h-sf-sp-full-hae-2deg-3mo",
    "ulc-ena-h-hf-sp-full-hae-4deg-3mo",
    "ulc-ena-h-sf-sp-full-hae-4deg-3mo",
    "ulc-ena-h-hf-sp-full-hae-6deg-3mo",
    "ulc-ena-h-sf-sp-full-hae-6deg-3mo",

    "u90-ena-h-hf-sp-full-hae-2deg-3mo",
    "u90-ena-h-sf-sp-full-hae-2deg-3mo",
    "u90-ena-h-hf-sp-full-hae-4deg-3mo",
    "u90-ena-h-sf-sp-full-hae-4deg-3mo",
    "u90-ena-h-hf-sp-full-hae-6deg-3mo",
    "u90-ena-h-sf-sp-full-hae-6deg-3mo",

    "u45-ena-h-hf-sp-full-hae-2deg-3mo",
    "u45-ena-h-sf-sp-full-hae-2deg-3mo",
    "u45-ena-h-hf-sp-full-hae-4deg-3mo",
    "u45-ena-h-sf-sp-full-hae-4deg-3mo",
    "u45-ena-h-hf-sp-full-hae-6deg-3mo",
    "u45-ena-h-sf-sp-full-hae-6deg-3mo",

    "ulc-ena-h-hf-sp-full-hae-2deg-6mo",
    "ulc-ena-h-sf-sp-full-hae-2deg-6mo",
    "ulc-ena-h-hf-sp-full-hae-4deg-6mo",
    "ulc-ena-h-sf-sp-full-hae-4deg-6mo",
    "ulc-ena-h-hf-sp-full-hae-6deg-6mo",
    "ulc-ena-h-sf-sp-full-hae-6deg-6mo",

    "u90-ena-h-hf-sp-full-hae-2deg-6mo",
    "u90-ena-h-sf-sp-full-hae-2deg-6mo",
    "u90-ena-h-hf-sp-full-hae-4deg-6mo",
    "u90-ena-h-sf-sp-full-hae-4deg-6mo",
    "u90-ena-h-hf-sp-full-hae-6deg-6mo",
    "u90-ena-h-sf-sp-full-hae-6deg-6mo",

    "u45-ena-h-hf-sp-full-hae-2deg-6mo",
    "u45-ena-h-sf-sp-full-hae-2deg-6mo",
    "u45-ena-h-hf-sp-full-hae-4deg-6mo",
    "u45-ena-h-sf-sp-full-hae-4deg-6mo",
    "u45-ena-h-hf-sp-full-hae-6deg-6mo",
    "u45-ena-h-sf-sp-full-hae-6deg-6mo",

    "ulc-ena-h-hf-sp-full-hae-2deg-1yr",
    "ulc-ena-h-sf-sp-full-hae-2deg-1yr",
    "ulc-ena-h-hf-sp-full-hae-4deg-1yr",
    "ulc-ena-h-sf-sp-full-hae-4deg-1yr",
    "ulc-ena-h-hf-sp-full-hae-6deg-1yr",
    "ulc-ena-h-sf-sp-full-hae-6deg-1yr",
    
    "u90-ena-h-hf-sp-full-hae-2deg-1yr",
    "u90-ena-h-sf-sp-full-hae-2deg-1yr",
    "u90-ena-h-hf-sp-full-hae-4deg-1yr",
    "u90-ena-h-sf-sp-full-hae-4deg-1yr",
    "u90-ena-h-hf-sp-full-hae-6deg-1yr",
    "u90-ena-h-sf-sp-full-hae-6deg-1yr",

    "u45-ena-h-hf-sp-full-hae-2deg-1yr",
    "u45-ena-h-sf-sp-full-hae-2deg-1yr",
    "u45-ena-h-hf-sp-full-hae-4deg-1yr",
    "u45-ena-h-sf-sp-full-hae-4deg-1yr",
    "u45-ena-h-hf-sp-full-hae-6deg-1yr",
    "u45-ena-h-sf-sp-full-hae-6deg-1yr",
]


class UltraInitializer(MapInitializer):
    def __init__(self):
        sp_query_result = imap_data_access.query(instrument='glows', data_level='l3e',
                                                 descriptor="survival-probability-ul", version="latest")
        self.glows_psets_by_repointing = {int(r["repointing"]): Path(r["file_path"]).name for r in sp_query_result}

        l2_query_result = imap_data_access.query(instrument="ultra", data_level="l2")
        l3_query_result = imap_data_access.query(instrument="ultra", data_level="l3")
        self._energy_bin_group_sizes_files = imap_data_access.query(
            table="ancillary",
            instrument="ultra",
            descriptor="l2-energy-bin-group-sizes",
            version="latest")
        super().__init__("ultra", l2_query_result, l3_query_result)

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(
            start_date=map_to_produce.input_metadata.start_date,
            end_date=map_to_produce.input_metadata.end_date,
            kernel_types=ULTRA_SP_SPICE_KERNELS
        )

    def _collect_glows_psets_by_repoint(self, descriptor: MapDescriptorParts) -> dict[int, str]:
        return self.glows_psets_by_repointing

    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected)]

    def _get_ancillary_files(self) -> list[str]:
        return [Path(f["file_path"]).name for f in self._energy_bin_group_sizes_files]
