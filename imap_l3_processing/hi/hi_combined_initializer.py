import dataclasses
from datetime import datetime, timedelta

import imap_data_access
from dateutil.relativedelta import relativedelta

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, Sensor, \
    map_descriptor_parts_to_string, SpinPhase
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce, MapInitializer
from imap_l3_processing.models import InputMetadata

HI_COMBINED_DESCRIPTORS = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
]

LENGTH_OF_YEAR_IN_DAYS = timedelta(days=365.25)


class HiCombinedInitializer(MapInitializer):
    input_map_descriptor_parts: dict[str, list[MapDescriptorParts]]
    input_map_filenames: dict[str, list[str]]

    def __init__(self, canonical_map_start_date: datetime = datetime(2026, 1, 1)):
        super().__init__('hi')

        l2_query_results = imap_data_access.query(instrument="hi", data_level="l2")
        l2_by_descriptor_and_start_date = self.get_latest_version_by_descriptor_and_start_date(l2_query_results)
        self.input_maps = dict(self.existing_l3_maps, **l2_by_descriptor_and_start_date)
        self.canonical_map_start_date = canonical_map_start_date

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        pass

    def get_maps_that_can_be_produced(self, input_descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps_to_produce = []
        map_descriptor: MapDescriptorParts = parse_map_descriptor(input_descriptor)

        hi90_ram_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi90, spin_phase=SpinPhase.RamOnly))
        hi45_ram_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi45, spin_phase=SpinPhase.RamOnly))

        hi90_anti_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi90, spin_phase=SpinPhase.AntiRamOnly))
        hi45_anti_descriptor = map_descriptor_parts_to_string(
            dataclasses.replace(map_descriptor, sensor=Sensor.Hi45, spin_phase=SpinPhase.AntiRamOnly))

        if hi90_ram_descriptor not in self.input_maps or hi90_anti_descriptor not in self.input_maps or hi45_ram_descriptor not in self.input_maps or hi45_anti_descriptor not in self.input_maps:
            return []

        hi90_start_dates = sorted(self.input_maps[hi90_ram_descriptor].keys())

        for start_date_str in hi90_start_dates:
            start_date = datetime.strptime(start_date_str, '%Y%m%d')

            input_files = [
                self.input_maps[hi90_ram_descriptor].get(start_date_str),
                self.input_maps[hi90_anti_descriptor].get(start_date_str),
                self.input_maps[hi45_ram_descriptor].get(start_date_str),
                self.input_maps[hi45_anti_descriptor].get(start_date_str),
            ]

            if not all(input_files):
                continue

            possible_maps_to_produce.append(PossibleMapToProduce(
                input_files=set(input_files),
                input_metadata=InputMetadata(
                    instrument='hi',
                    data_level='l3',
                    start_date=start_date,
                    end_date=start_date + relativedelta(months=+12),
                    version='v001',
                    descriptor=input_descriptor,
                    repointing=None
                )
            ))

        return possible_maps_to_produce
