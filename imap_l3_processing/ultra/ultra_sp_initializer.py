import dataclasses
from datetime import datetime
from pathlib import Path

import imap_data_access

from imap_l3_processing.glows.descriptors import GLOWS_L3E_ULTRA_HF_DESCRIPTOR, GLOWS_L3E_ULTRA_SF_DESCRIPTOR
from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, SurvivalCorrection, Sensor, ReferenceFrame, \
    map_descriptor_parts_to_string
from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.maps.sp_map_initializer import SPMapInitializer
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import furnish_spice_metakernel, SpiceKernelTypes

ULTRA_SP_SPICE_KERNELS = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
]

ULTRA_45_DESCRIPTORS = [
    "u45-ena-h-hf-sp-full-hae-2deg-3mo",
    "u45-ena-h-sf-sp-full-hae-2deg-3mo",
    "u45-ena-h-hf-sp-full-hae-4deg-3mo",
    "u45-ena-h-sf-sp-full-hae-4deg-3mo",
    "u45-ena-h-hf-sp-full-hae-6deg-3mo",
    "u45-ena-h-sf-sp-full-hae-6deg-3mo",

    "u45-ena-h-hf-sp-full-hae-2deg-6mo",
    "u45-ena-h-sf-sp-full-hae-2deg-6mo",
    "u45-ena-h-hf-sp-full-hae-4deg-6mo",
    "u45-ena-h-sf-sp-full-hae-4deg-6mo",
    "u45-ena-h-hf-sp-full-hae-6deg-6mo",
    "u45-ena-h-sf-sp-full-hae-6deg-6mo",

    "u45-ena-h-hf-sp-full-hae-2deg-1yr",
    "u45-ena-h-sf-sp-full-hae-2deg-1yr",
    "u45-ena-h-hf-sp-full-hae-4deg-1yr",
    "u45-ena-h-sf-sp-full-hae-4deg-1yr",
    "u45-ena-h-hf-sp-full-hae-6deg-1yr",
    "u45-ena-h-sf-sp-full-hae-6deg-1yr",
]

ULTRA_90_DESCRIPTORS = [
    "u90-ena-h-hf-sp-full-hae-2deg-3mo",
    "u90-ena-h-sf-sp-full-hae-2deg-3mo",
    "u90-ena-h-hf-sp-full-hae-4deg-3mo",
    "u90-ena-h-sf-sp-full-hae-4deg-3mo",
    "u90-ena-h-hf-sp-full-hae-6deg-3mo",
    "u90-ena-h-sf-sp-full-hae-6deg-3mo",

    "u90-ena-h-hf-sp-full-hae-2deg-6mo",
    "u90-ena-h-sf-sp-full-hae-2deg-6mo",
    "u90-ena-h-hf-sp-full-hae-4deg-6mo",
    "u90-ena-h-sf-sp-full-hae-4deg-6mo",
    "u90-ena-h-hf-sp-full-hae-6deg-6mo",
    "u90-ena-h-sf-sp-full-hae-6deg-6mo",

    "u90-ena-h-hf-sp-full-hae-2deg-1yr",
    "u90-ena-h-sf-sp-full-hae-2deg-1yr",
    "u90-ena-h-hf-sp-full-hae-4deg-1yr",
    "u90-ena-h-sf-sp-full-hae-4deg-1yr",
    "u90-ena-h-hf-sp-full-hae-6deg-1yr",
    "u90-ena-h-sf-sp-full-hae-6deg-1yr",
]

ULTRA_COMBINED_SP_DESCRIPTORS = [
    "ulc-ena-h-hf-sp-full-hae-2deg-3mo",
    "ulc-ena-h-hf-sp-full-hae-4deg-3mo",
    "ulc-ena-h-hf-sp-full-hae-6deg-3mo",

    "ulc-ena-h-hf-sp-full-hae-2deg-6mo",
    "ulc-ena-h-hf-sp-full-hae-4deg-6mo",
    "ulc-ena-h-hf-sp-full-hae-6deg-6mo",

    "ulc-ena-h-hf-sp-full-hae-2deg-1yr",
    "ulc-ena-h-hf-sp-full-hae-4deg-1yr",
    "ulc-ena-h-hf-sp-full-hae-6deg-1yr",
]

ULTRA_COMBINED_NSP_DESCRIPTORS = [
    "ulc-ena-h-hf-nsp-full-hae-2deg-3mo",
    "ulc-ena-h-hf-nsp-full-hae-4deg-3mo",
    "ulc-ena-h-hf-nsp-full-hae-6deg-3mo",

    "ulc-ena-h-hf-nsp-full-hae-2deg-6mo",
    "ulc-ena-h-hf-nsp-full-hae-4deg-6mo",
    "ulc-ena-h-hf-nsp-full-hae-6deg-6mo",

    "ulc-ena-h-hf-nsp-full-hae-2deg-1yr",
    "ulc-ena-h-hf-nsp-full-hae-4deg-1yr",
    "ulc-ena-h-hf-nsp-full-hae-6deg-1yr",
]


class UltraSPInitializer(SPMapInitializer):
    def __init__(self):
        sf_sp_query_result = imap_data_access.query(instrument='glows', data_level='l3e',
                                                    descriptor=GLOWS_L3E_ULTRA_SF_DESCRIPTOR, version="latest")
        self.sf_glows_psets_by_repointing = {int(r["repointing"]): Path(r["file_path"]).name for r in
                                             sf_sp_query_result}
        hf_sp_query_result = imap_data_access.query(instrument='glows', data_level='l3e',
                                                    descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR, version="latest")
        self.hf_glows_psets_by_repointing = {int(r["repointing"]): Path(r["file_path"]).name for r in
                                             hf_sp_query_result}

        l2_query_result = imap_data_access.query(instrument="ultra", data_level="l2")
        self._energy_bin_group_sizes_files = imap_data_access.query(
            instrument="ultra",
            table="ancillary",
            descriptor="l2-energy-bin-group-sizes",
            version="latest")

        super().__init__("ultra", l2_query_result)

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(
            start_date=map_to_produce.input_metadata.start_date,
            end_date=map_to_produce.input_metadata.end_date,
            kernel_types=ULTRA_SP_SPICE_KERNELS
        )

    def _collect_glows_psets_by_repoint(self, descriptor: MapDescriptorParts) -> dict[int, str]:
        if descriptor.reference_frame == ReferenceFrame.Heliospheric:
            return self.hf_glows_psets_by_repointing
        elif descriptor.reference_frame == ReferenceFrame.Spacecraft:
            return self.sf_glows_psets_by_repointing
        else:
            raise NotImplementedError("Reference frame should be either Spacecraft or Heliospheric")

    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        if descriptor.sensor == Sensor.UltraCombined:
            return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                        sensor=Sensor.Ultra45),
                    dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected,
                                        sensor=Sensor.Ultra90)]

        return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected)]

    def _get_ancillary_files(self) -> list[str]:
        return [Path(f["file_path"]).name for f in self._energy_bin_group_sizes_files]


class UltraCombinedNSPInitializer(MapInitializer):

    def __init__(self):
        l2_ultra_maps_query_results = imap_data_access.query(instrument="ultra", data_level="l2")
        self.l2_maps_by_descriptor_and_start = self.get_latest_version_by_descriptor_and_start_date(
            l2_ultra_maps_query_results)
        super().__init__("ultra")

    @staticmethod
    def _get_l2_dependencies(descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        return [
            dataclasses.replace(descriptor, sensor=Sensor.Ultra45),
            dataclasses.replace(descriptor, sensor=Sensor.Ultra90),
        ]

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(
            start_date=map_to_produce.input_metadata.start_date,
            end_date=map_to_produce.input_metadata.end_date,
            kernel_types=ULTRA_SP_SPICE_KERNELS
        )

    def get_maps_that_can_be_produced(self, descriptor: MapDescriptorParts) -> list[PossibleMapToProduce]:
        input_map_descriptors = [map_descriptor_parts_to_string(desc) for desc in self._get_l2_dependencies(descriptor)]

        first_input_descriptor = input_map_descriptors[0]
        possible_start_dates = self.l2_maps_by_descriptor_and_start[first_input_descriptor].keys()

        possible_maps_to_produce = []
        for str_start_date in possible_start_dates:
            input_map_filenames = []
            for input_descriptor in input_map_descriptors:
                if str_start_date in self.l2_maps_by_descriptor_and_start[input_descriptor]:
                    input_map_filenames.append(self.l2_maps_by_descriptor_and_start[input_descriptor][str_start_date])

            if len(input_map_filenames) != len(input_map_descriptors):
                continue

            pset_filenames = [l1c for l2 in input_map_filenames for l1c in self.get_l1c_parents_from_map(l2)]

            possible_maps_to_produce.append(PossibleMapToProduce(
                input_files=set(pset_filenames + input_map_filenames),
                input_metadata=InputMetadata(
                    instrument='ultra',
                    data_level='l3',
                    start_date=datetime.strptime(str_start_date, "%Y%m%d"),
                    end_date=None,
                    version='v001',
                    descriptor=map_descriptor_parts_to_string(descriptor),
                )
            ))

        return possible_maps_to_produce
