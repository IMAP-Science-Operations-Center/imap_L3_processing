import dataclasses
from datetime import datetime

import imap_data_access
from dateutil.relativedelta import relativedelta

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, Sensor, \
    map_descriptor_parts_to_string
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce, MapInitializer
from imap_l3_processing.models import InputMetadata

HI_COMBINED_DESCRIPTORS = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
]


class HiCombinedInitializer(MapInitializer):
    input_map_descriptor_parts: dict[str, list[MapDescriptorParts]]
    input_map_filenames: dict[str, list[str]]

    def __init__(self):
        super().__init__('hi')

        l2_query_results = imap_data_access.query(instrument="hi", data_level="l2")
        l2_by_descriptor_and_start_date = self.get_latest_version_by_descriptor_and_start_date(l2_query_results)
        self.input_maps = dict(self.existing_l3_maps, **l2_by_descriptor_and_start_date)

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        pass

    def get_maps_that_can_be_produced(self, input_descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps_to_produce = []
        map_descriptor: MapDescriptorParts = parse_map_descriptor(input_descriptor)

        hi90_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi90, duration="6mo"))
        hi45_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi45, duration="6mo"))

        if hi90_descriptor not in self.input_maps or hi45_descriptor not in self.input_maps:
            return []

        hi90_start_dates = sorted(self.input_maps[hi90_descriptor].keys())
        for i in range(0, len(hi90_start_dates), 2):
            first_6mo_start_date_str = hi90_start_dates[i]
            first_6mo_start_date = datetime.strptime(first_6mo_start_date_str, '%Y%m%d')

            if i + 1 < len(hi90_start_dates):
                six_months_later = first_6mo_start_date + relativedelta(months=+6)
                next_map_start_date_str = hi90_start_dates[i + 1]
                next_map_start_date = datetime.strptime(next_map_start_date_str, "%Y%m%d")

                if six_months_later == next_map_start_date:
                    input_files = [
                        self.input_maps[hi90_descriptor].get(first_6mo_start_date_str),
                        self.input_maps[hi90_descriptor].get(next_map_start_date_str),
                        self.input_maps[hi45_descriptor].get(first_6mo_start_date_str),
                        self.input_maps[hi45_descriptor].get(next_map_start_date_str),
                    ]

                    if not all(input_files):
                        continue

                    possible_maps_to_produce.append(PossibleMapToProduce(
                        input_files=set(input_files),
                        input_metadata=InputMetadata(
                            instrument='hi',
                            data_level='l3',
                            start_date=first_6mo_start_date,
                            end_date=first_6mo_start_date + relativedelta(months=+12),
                            version='v001',
                            descriptor=input_descriptor,
                            repointing=None
                        )
                    ))

        return possible_maps_to_produce
