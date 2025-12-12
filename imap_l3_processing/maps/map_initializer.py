import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imap_data_access
from imap_data_access import ScienceFilePath, ImapFilePath, ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import read_cdf_parents

logger = logging.getLogger(__name__)


@dataclass
class PossibleMapToProduce:
    input_files: set[str]
    input_metadata: InputMetadata

    @property
    def processing_input_collection(self) -> ProcessingInputCollection:
        return ProcessingInputCollection(*[generate_imap_input(file_path) for file_path in sorted(self.input_files)])


class MapInitializer(abc.ABC):
    def __init__(self, instrument: str):
        l3_query_results = imap_data_access.query(instrument=instrument, data_level='l3')
        self.existing_l3_maps = self.get_latest_version_by_descriptor_and_start_date(l3_query_results)
        self.instrument = instrument

    @abc.abstractmethod
    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_maps_that_can_be_produced(self, l3_descriptor: str):
        raise NotImplementedError()

    def get_maps_that_should_be_produced(self, descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps = self.get_maps_that_can_be_produced(descriptor)

        maps_to_make = []
        for possible_map in possible_maps:
            start_time = possible_map.input_metadata.start_date.strftime("%Y%m%d")
            if start_dates_for_l3_descriptor := self.existing_l3_maps.get(descriptor):
                if l3_result := start_dates_for_l3_descriptor.get(start_time):
                    existing_parents = read_cdf_parents(l3_result)
                    if possible_map.input_files.issubset(existing_parents):
                        continue
                    new_version = int(ScienceFilePath(l3_result).version[1:]) + 1
                    possible_map.input_metadata.version = f'v{new_version:03}'
            maps_to_make.append(possible_map)
        return maps_to_make

    @staticmethod
    def get_l1c_parents_from_map(map_file_path: str) -> list[str]:
        l1c_names = []
        for l1 in read_cdf_parents(map_file_path):
            try:
                l1c_science_file_path = ScienceFilePath(l1)
                if l1c_science_file_path.data_level == 'l1c':
                    l1c_names.append(l1c_science_file_path.filename.name)
            except ImapFilePath.InvalidImapFileError:
                continue
        return l1c_names

    @staticmethod
    def get_latest_version_by_descriptor_and_start_date(query_results: list[dict]) -> defaultdict:
        qr_by_descriptor_and_start_date = defaultdict(list)
        for qr in query_results:
            qr_by_descriptor_and_start_date[(qr['descriptor'], qr["start_date"])].append(qr)

        file_paths_by_descriptor = defaultdict(dict)
        for (descriptor, start_date), query_results in qr_by_descriptor_and_start_date.items():
            highest_version_qr = max(query_results, key=lambda x: x['version'])
            file_paths_by_descriptor[descriptor][start_date] = Path(highest_version_qr['file_path']).name

        return file_paths_by_descriptor
