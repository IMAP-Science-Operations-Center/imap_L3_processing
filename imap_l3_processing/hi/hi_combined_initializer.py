from datetime import datetime

from dateutil.relativedelta import relativedelta

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, ReferenceFrame, \
    SpinPhase, parse_map_descriptor
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce, MapInitializer
from imap_l3_processing.models import InputMetadata

HI_COMBINED_DESCRIPTORS = [
    "hic-ena-h-hf-nsp-full-hae-6deg-1yr",
    "hic-ena-h-hf-sp-full-hae-6deg-1yr",
    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
    "hic-ena-h-hf-sp-full-hae-4deg-1yr",
]


class HiCombinedL3Initializer(MapInitializer):
    input_map_descriptor_parts: dict[str, list[MapDescriptorParts]]
    input_map_filenames: dict[str, list[str]]
    existing_combined_maps: dict[str, dict[str, dict]]

    def __init__(self):
        super().__init__('hi')

        self.input_map_descriptor_parts = {}
        self.input_map_filenames = {}

        for descriptor in self.existing_l3_maps.keys():
            for start_date in self.existing_l3_maps[descriptor].keys():
                descriptor_parts = parse_map_descriptor(descriptor)
                maps_by_start_date = self.existing_l3_maps[descriptor]
                match descriptor_parts:
                    case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                            duration='6mo'):
                        if start_date not in self.input_map_descriptor_parts:
                            self.input_map_descriptor_parts[start_date] = []
                            self.input_map_filenames[start_date] = []

                        self.input_map_descriptor_parts[start_date].append(descriptor_parts)
                        self.input_map_filenames[start_date].append(maps_by_start_date[start_date])

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        pass

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
