import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from imap_data_access import ScienceFilePath, ImapFilePath, ProcessingInputCollection, ScienceInput

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, parse_map_descriptor, \
    map_descriptor_parts_to_string
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
    def __init__(self, instrument: str, l2_query_results: list[dict[str, str]], l3_query_results: list[dict[str, str]]):
        self.instrument = instrument
        self.l2_file_paths_by_descriptor = get_latest_version_by_descriptor_and_start_date(l2_query_results)
        self.existing_l3_maps = get_latest_version_by_descriptor_and_start_date(l3_query_results)

    @abc.abstractmethod
    def furnish_spice_dependencies(self, map_to_produce: PossibleMapToProduce):
        raise NotImplementedError()

    @abc.abstractmethod
    def _collect_glows_psets_by_repoint(self, descriptor: MapDescriptorParts) -> dict[int, str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        raise NotImplementedError()

    @staticmethod
    def get_duration_from_map_descriptor(descriptor: MapDescriptorParts) -> timedelta:
        match descriptor:
            case MapDescriptorParts(duration="1mo"):
                return timedelta(days=365.25) / 12
            case MapDescriptorParts(duration="3mo"):
                return timedelta(days=365.25) / 4
            case MapDescriptorParts(duration="6mo"):
                return timedelta(days=365.25) / 2
            case MapDescriptorParts(duration="1yr" | "12mo"):
                return timedelta(days=365.25)
            case _:
                raise ValueError(f"Expected a duration in the map descriptor, got: {descriptor} (e.g., '1mo', '3mo')")

    def get_maps_that_can_be_produced(self, l3_descriptor: str) -> list[PossibleMapToProduce]:
        l3_descriptor_parts = parse_map_descriptor(l3_descriptor)
        map_duration = self.get_duration_from_map_descriptor(l3_descriptor_parts)

        glows_file_by_repointing = self._collect_glows_psets_by_repoint(l3_descriptor_parts)
        if len(glows_file_by_repointing) == 0:
            logger.info(f"No GLOWS data available for descriptor {l3_descriptor}, no maps will be produced!")
            return []
        glows_start_dates = [ScienceFilePath(glows_path).start_date for glows_path in glows_file_by_repointing.values()]
        glows_data_end_date = datetime.strptime(max(glows_start_dates), "%Y%m%d")

        l2_descriptors = self._get_l2_dependencies(l3_descriptor_parts)
        l2_descriptor_strs = [map_descriptor_parts_to_string(parts) for parts in l2_descriptors]
        assert l2_descriptor_strs, f"Expected at least one L2 dependency for l3 map: {l3_descriptor}"

        possible_start_dates = set(self.l2_file_paths_by_descriptor[l2_descriptor_strs[0]].keys())
        for l2_descriptor in l2_descriptor_strs[1:]:
            possible_start_dates.intersection_update(self.l2_file_paths_by_descriptor[l2_descriptor].keys())

        possible_maps = []
        for str_start_date in sorted(list(possible_start_dates)):
            start_date = datetime.strptime(str_start_date, "%Y%m%d")
            if start_date + map_duration > glows_data_end_date:
                logger.info(f"Not enough GLOWS data to produce map {l3_descriptor} {str_start_date}")
                continue

            l2_file_paths = []
            for l2_descriptor in l2_descriptor_strs:
                l2_file_paths.append(self.l2_file_paths_by_descriptor[l2_descriptor][str_start_date])

            l1c_names = self.get_l1c_parents_from_map(l2_file_paths[0])
            mismatch = False
            for l2_path in l2_file_paths[1:]:
                if set(self.get_l1c_parents_from_map(l2_path)) != set(l1c_names):
                    mismatch = True
                    break

            if mismatch:
                logger.warning(
                    f"Expected all input maps to be created from the same pointing sets! l2_file_paths: "
                    f"{', '.join(l2_file_paths)}"
                )
                continue

            l1c_repointings = [ScienceFilePath(l1c).repointing for l1c in l1c_names]

            glows_files = [glows_file_by_repointing[repoint] for repoint in l1c_repointings if
                           repoint in glows_file_by_repointing]

            if len(glows_files) > 0:
                input_metadata = InputMetadata(instrument=self.instrument, data_level='l3', start_date=start_date,
                                               end_date=start_date,
                                               version='v001', descriptor=l3_descriptor)

                possible_map_to_produce = PossibleMapToProduce(
                    input_files=set(l2_file_paths + glows_files + l1c_names),
                    input_metadata=input_metadata
                )
                possible_maps.append(possible_map_to_produce)
        return possible_maps

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

def get_latest_version_by_descriptor_and_start_date(query_results: list[dict]) -> defaultdict:
    qr_by_descriptor_and_start_date = defaultdict(list)
    for qr in query_results:
        qr_by_descriptor_and_start_date[(qr['descriptor'], qr["start_date"])].append(qr)

    file_paths_by_descriptor = defaultdict(dict)
    for (descriptor, start_date), query_results in qr_by_descriptor_and_start_date.items():
        highest_version_qr = max(query_results, key=lambda x: x['version'])
        file_paths_by_descriptor[descriptor][start_date] = Path(highest_version_qr['file_path']).name

    return file_paths_by_descriptor
