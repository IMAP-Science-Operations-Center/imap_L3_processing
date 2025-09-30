import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from imap_data_access import ScienceFilePath, ImapFilePath, ProcessingInputCollection, ScienceInput

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import read_cdf_parents

logger = logging.getLogger(__name__)


@dataclass
class PossibleMapToProduce:
    input_files: set[str]
    input_metadata: InputMetadata

    @property
    def processing_input_collection(self) -> ProcessingInputCollection:
        return ProcessingInputCollection(*[ScienceInput(file_path) for file_path in list(self.input_files)])


class MapInitializer(abc.ABC):
    def __init__(self, l2_query_results: list[dict[str, str]], l3_query_results: list[dict[str, str]]):
        self.l2_file_paths_by_descriptor = defaultdict(dict)
        for result in l2_query_results:
            logger.info(f"l2 file in MapInitializer __init__: {result['file_path']}")
            self.l2_file_paths_by_descriptor[result['descriptor']][result["start_date"]] = result['file_path']

        self.existing_l3_maps = {(qr["descriptor"], qr["start_date"]): qr["file_path"] for qr in l3_query_results}

    @abc.abstractmethod
    def _collect_glows_psets_by_repoint(self, descriptor: str) -> dict[int, str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_l2_dependencies(self, descriptor: str) -> list[str]:
        raise NotImplementedError()

    def get_maps_that_can_be_produced(self, l3_descriptor: str) -> list[PossibleMapToProduce]:
        l2_descriptors = self._get_l2_dependencies(l3_descriptor)
        assert l2_descriptors, f"Expected at least one L2 dependency for l3 map: {l3_descriptor}"

        glows_file_by_repointing = self._collect_glows_psets_by_repoint(l3_descriptor)

        possible_start_dates = set(self.l2_file_paths_by_descriptor[l2_descriptors[0]].keys())
        for l2_descriptor in l2_descriptors[1:]:
            possible_start_dates.intersection_update(self.l2_file_paths_by_descriptor[l2_descriptor].keys())

        possible_maps = []
        for str_start_date in sorted(list(possible_start_dates)):
            l2_files_paths = []
            l1c_names = []
            for l2_descriptor in l2_descriptors:
                l2_file_path = self.l2_file_paths_by_descriptor[l2_descriptor][str_start_date]
                l2_files_paths.append(l2_file_path)
                l1c_names.extend(read_cdf_parents(l2_file_path))

            start_date = datetime.strptime(str_start_date, "%Y%m%d")

            l1c_repointings = []
            for l1 in l1c_names:
                try:
                    l1c_repointings.append(ScienceFilePath(l1).repointing)
                except ImapFilePath.InvalidImapFileError:
                    continue

            glows_files = [glows_file_by_repointing[repoint] for repoint in l1c_repointings if
                           repoint in glows_file_by_repointing]

            if len(glows_files) > 0:
                input_metadata = InputMetadata(instrument='hi', data_level='l3', start_date=start_date,
                                               end_date=start_date,
                                               version='v001', descriptor=l3_descriptor)

                possible_map_to_produce = PossibleMapToProduce(
                    input_files=set(l2_files_paths + glows_files),
                    input_metadata=input_metadata
                )
                possible_maps.append(possible_map_to_produce)
        return possible_maps

    def get_maps_that_should_be_produced(self, descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps = self.get_maps_that_can_be_produced(descriptor)

        maps_to_make = []
        for map in possible_maps:
            start_time = map.input_metadata.start_date.strftime("%Y%m%d")
            if l3_result := self.existing_l3_maps.get((descriptor, start_time)):
                if map.input_files.issubset(read_cdf_parents(l3_result)):
                    continue
                new_version = int(ScienceFilePath(l3_result).version[1:]) + 1
                map.input_metadata.version = f'v{new_version:03}'
            maps_to_make.append(map)
        return maps_to_make
