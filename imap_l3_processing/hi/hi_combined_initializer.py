from datetime import datetime

import imap_data_access
from dateutil.relativedelta import relativedelta

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, ReferenceFrame, \
    SpinPhase, parse_map_descriptor, PixelSize, SurvivalCorrection
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

        self.l3_6deg_input_map_filenames = {}
        self.l3_4deg_input_map_filenames = {}

        self.l2_6deg_input_map_filenames = {}
        self.l2_4deg_input_map_filenames = {}

        self.input_map_data = {
            (SurvivalCorrection.SurvivalCorrected.value, PixelSize.SixDegrees.value): self.l3_6deg_input_map_filenames,
            (SurvivalCorrection.NotSurvivalCorrected.value,
             PixelSize.SixDegrees.value): self.l2_6deg_input_map_filenames,
            (SurvivalCorrection.SurvivalCorrected.value, PixelSize.FourDegrees.value): self.l3_4deg_input_map_filenames,
            (SurvivalCorrection.NotSurvivalCorrected.value,
             PixelSize.FourDegrees.value): self.l2_4deg_input_map_filenames
        }

        l2_query_results = imap_data_access.query(instrument="hi", data_level="l2")
        self.existing_l2_maps = self.get_latest_version_by_descriptor_and_start_date(l2_query_results)

        for descriptor in self.existing_l3_maps.keys():
            for start_date in self.existing_l3_maps[descriptor].keys():
                descriptor_parts = parse_map_descriptor(descriptor)
                maps_by_start_date = self.existing_l3_maps[descriptor]
                match descriptor_parts:
                    case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                            duration='6mo', grid=PixelSize.SixDegrees):
                        if start_date not in self.l3_6deg_input_map_filenames:
                            self.l3_6deg_input_map_filenames[start_date] = []

                        self.l3_6deg_input_map_filenames[start_date].append(maps_by_start_date[start_date])

                match descriptor_parts:
                    case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                            duration='6mo', grid=PixelSize.FourDegrees):
                        if start_date not in self.l3_4deg_input_map_filenames:
                            self.l3_4deg_input_map_filenames[start_date] = []

                        self.l3_4deg_input_map_filenames[start_date].append(maps_by_start_date[start_date])

        for descriptor in self.existing_l2_maps.keys():
            for start_date in self.existing_l2_maps[descriptor].keys():
                descriptor_parts = parse_map_descriptor(descriptor)
                maps_by_start_date = self.existing_l2_maps[descriptor]
                match descriptor_parts:
                    case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                            duration='6mo', grid=PixelSize.SixDegrees):
                        if start_date not in self.l2_6deg_input_map_filenames:
                            self.l2_6deg_input_map_filenames[start_date] = []

                        self.l2_6deg_input_map_filenames[start_date].append(maps_by_start_date[start_date])

                    case MapDescriptorParts(reference_frame=ReferenceFrame.Heliospheric, spin_phase=SpinPhase.FullSpin,
                                            duration='6mo', grid=PixelSize.FourDegrees):
                        if start_date not in self.l2_4deg_input_map_filenames:
                            self.l2_4deg_input_map_filenames[start_date] = []

                        self.l2_4deg_input_map_filenames[start_date].append(maps_by_start_date[start_date])

    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        pass

    def get_maps_that_can_be_produced(self, input_descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps_to_produce = []
        map_descriptor: MapDescriptorParts = parse_map_descriptor(input_descriptor)

        input_map_filenames = self.input_map_data.get(
            (map_descriptor.survival_correction.value, map_descriptor.grid.value))

        sorted_keys = sorted(input_map_filenames.keys())

        for i in range(0, len(sorted_keys), 2):

            start_date_str = sorted_keys[i]

            start_date = datetime.strptime(start_date_str, '%Y%m%d')
            if i + 1 < len(sorted_keys):
                six_months_later = start_date + relativedelta(months=+6)
                next_date_str = sorted_keys[i + 1]
                next_date = datetime.strptime(next_date_str, '%Y%m%d')
                if six_months_later == next_date:
                    input_files = input_map_filenames[start_date_str] + input_map_filenames[
                        next_date_str]

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
