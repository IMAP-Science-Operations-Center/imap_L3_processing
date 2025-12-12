import dataclasses
from datetime import datetime
from pathlib import Path

import imap_data_access

from imap_l3_processing.maps.map_descriptors import Sensor, map_descriptor_parts_to_string, \
    parse_map_descriptor, get_duration_from_map_descriptor
from imap_l3_processing.maps.map_initializer import MapInitializer, PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.ultra_sp_initializer import ULTRA_SP_SPICE_KERNELS
from imap_l3_processing.utils import furnish_spice_metakernel

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

class UltraCombinedNSPInitializer(MapInitializer):
    def __init__(self):
        l2_ultra_maps_query_results = imap_data_access.query(instrument="ultra", data_level="l2")
        self.l2_maps_by_descriptor_and_start = self.get_latest_version_by_descriptor_and_start_date(
            l2_ultra_maps_query_results)
        energy_bin_group_sizes_files = imap_data_access.query(
            instrument="ultra",
            table="ancillary",
            descriptor="l2-energy-bin-group-sizes",
            version="latest")
        self.ancillary_dependencies = [Path(qr['file_path']).name for qr in energy_bin_group_sizes_files]
        super().__init__("ultra")

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        furnish_spice_metakernel(
            start_date=map_to_produce.input_metadata.start_date,
            end_date=map_to_produce.input_metadata.end_date,
            kernel_types=ULTRA_SP_SPICE_KERNELS
        )

    def get_maps_that_can_be_produced(self, descriptor: str) -> list[PossibleMapToProduce]:
        descriptor_parts = parse_map_descriptor(descriptor)
        if descriptor_parts is None:
            raise ValueError(f"Invalid map descriptor: {descriptor}")

        dependency_descriptors = [
            dataclasses.replace(descriptor_parts, sensor=Sensor.Ultra45),
            dataclasses.replace(descriptor_parts, sensor=Sensor.Ultra90),
        ]
        input_map_descriptors = [map_descriptor_parts_to_string(parts) for parts in dependency_descriptors]

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

            start_date = datetime.strptime(str_start_date, "%Y%m%d")
            possible_maps_to_produce.append(PossibleMapToProduce(
                input_files=set(pset_filenames + input_map_filenames + self.ancillary_dependencies),
                input_metadata=InputMetadata(
                    instrument='ultra',
                    data_level='l3',
                    start_date=start_date,
                    end_date=start_date + get_duration_from_map_descriptor(descriptor_parts),
                    version='v001',
                    descriptor=descriptor,
                )
            ))

        return possible_maps_to_produce
