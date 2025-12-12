from datetime import datetime

import imap_data_access
from dateutil.relativedelta import relativedelta
from imap_data_access import ScienceFilePath

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, ReferenceFrame, \
    SpinPhase, parse_map_descriptor, Sensor
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import read_cdf_parents

HI_COMBINED_DESCRIPTORS = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
]


class HiCombinedL3Initializer():
    input_map_descriptor_parts: dict[str, list[MapDescriptorParts]]
    input_map_filenames: dict[str, list[str]]
    existing_combined_maps: dict[str, dict[str, str]]

    def __init__(self):
        self.input_map_descriptor_parts = {}
        self.input_map_filenames = {}
        self.existing_combined_maps = {}

        l3_query_results = imap_data_access.query(instrument='hi', data_level='l3')

        for result in l3_query_results:
            descriptor_parts = parse_map_descriptor(result['descriptor'])
            match descriptor_parts:
                case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                        duration='6mo'):
                    if result['start_date'] not in self.input_map_descriptor_parts:
                        self.input_map_descriptor_parts[result['start_date']] = []
                        self.input_map_filenames[result['start_date']] = []

                    self.input_map_descriptor_parts[result['start_date']].append(descriptor_parts)
                    self.input_map_filenames[result['start_date']].append(result['file_path'].split('/')[-1])

                case MapDescriptorParts(sensor=Sensor.HiCombined):
                    if result['start_date'] not in self.existing_combined_maps:
                        self.existing_combined_maps[result['start_date']] = {}
                    self.existing_combined_maps[result['start_date']][result['descriptor']] = \
                        result['file_path'].split('/')[-1]

    def get_maps_that_should_be_produced(self, input_descriptor) -> list[PossibleMapToProduce]:
        possible_maps = self.get_maps_that_can_be_produced(input_descriptor)

        maps_to_make = []
        for possible_map in possible_maps:
            start_time = possible_map.input_metadata.start_date.strftime('%Y%m%d')
            if start_date_for_l3_descriptor := self.existing_combined_maps.get(start_time):
                if l3_result := start_date_for_l3_descriptor.get(input_descriptor):
                    existing_parents = read_cdf_parents(l3_result)
                    if possible_map.input_files.issubset(existing_parents):
                        continue
                    new_version = int(ScienceFilePath(l3_result).version[1:]) + 1
                    possible_map.input_metadata.version = f'v{new_version:03}'
            maps_to_make.append(possible_map)
        return maps_to_make

    def get_maps_that_can_be_produced(self, input_descriptor: str) -> list[PossibleMapToProduce]:
        sorted_keys = sorted(self.input_map_filenames.keys())

        possible_maps_to_produce = []

        for i in range(0, len(sorted_keys), 2):

            start_date_str = sorted_keys[i]

            start_date = datetime.strptime(start_date_str, '%Y%m%d')
            if i + 1 < len(sorted_keys):
                six_months_later = start_date + relativedelta(months=+6)
                next_date_str = sorted_keys[i + 1]
                next_date = datetime.strptime(next_date_str, '%Y%m%d')
                if six_months_later == next_date:
                    input_files = self.input_map_filenames[start_date_str] + self.input_map_filenames[next_date_str]

                    possible_maps_to_produce.append(PossibleMapToProduce(
                        input_files=set(input_files),
                        input_metadata=InputMetadata(
                            instrument='hi',
                            data_level='l3',
                            start_date=start_date,
                            end_date=start_date + relativedelta(months=+12),
                            version='v001',
                            descriptor='hic-ena-h-hf-sp-full-hae-6deg-1yr',
                            repointing=None
                        )
                    ))

        return possible_maps_to_produce
