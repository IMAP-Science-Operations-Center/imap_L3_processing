import abc
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from imap_data_access import ScienceFilePath, ImapFilePath

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import read_cdf_parents


@dataclass
class PossibleMapToProduce:
    input_files: set[str]
    input_metadata: InputMetadata


class MapInitializer(abc.ABC):
    def __init__(self, l2_query_results: list[dict[str, str]], l3_query_results: list[dict[str, str]]):
        self.l2_file_paths_by_descriptor = defaultdict(list)
        for result in l2_query_results:
            self.l2_file_paths_by_descriptor[result['descriptor']].append(result['file_path'])
        self.existing_l3_maps = {(qr["descriptor"], qr["start_date"]): qr["file_path"] for qr in l3_query_results}

    @abc.abstractmethod
    def _collect_glows_psets_by_repoint(self, descriptor: str) -> dict[int, str]:
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_dependencies():
        raise NotImplementedError()

    def get_maps_that_can_be_produced(self, l3_descriptor: str) -> list[PossibleMapToProduce]:
        l2_descriptor = l3_descriptor.replace('-sp-', '-nsp-')

        glows_file_by_repointing = self._collect_glows_psets_by_repoint(l3_descriptor)

        possible_maps = []
        for l2_file_path in self.l2_file_paths_by_descriptor[l2_descriptor]:
            start_date = datetime.strptime(ScienceFilePath(l2_file_path).start_date, "%Y%m%d")
            l1c_names = read_cdf_parents(l2_file_path)

            l1c_repointings = []
            for l1 in l1c_names:
                try:
                    l1c_repointings.append(ScienceFilePath(l1).repointing)
                except ImapFilePath.InvalidImapFileError:
                    continue

            glows_files = [glows_file_by_repointing[repoint] for repoint in l1c_repointings if repoint in glows_file_by_repointing]

            if len(glows_files) > 0:
                input_metadata = InputMetadata(instrument='hi', data_level='l3', start_date=start_date, end_date=start_date,
                                               version='v001', descriptor=l3_descriptor)

                possible_map_to_produce = PossibleMapToProduce(
                    input_files=set([l2_file_path] + glows_files),
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