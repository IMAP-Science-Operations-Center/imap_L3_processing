import logging
from dataclasses import dataclass
from datetime import datetime

import imap_data_access
from fontTools.config import Config
from imap_data_access.file_validation import ImapFilePath
from imap_data_access.file_validation import ScienceFilePath

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import read_cdf_parents

logger = logging.getLogger(__name__)


@dataclass
class PossibleMapToProduce:
    input_files: set[str]
    input_metadata: InputMetadata


class HiL3Initializer:
    def __init__(self):
        sp_hi45_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor=f'survival-probability-hi-45',
            version='latest'
        )
        self.glows_hi45_file_by_repoint = {r["repointing"]: r["file_path"] for r in sp_hi45_query_result}

        sp_hi90_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor=f'survival-probability-hi-90',
            version='latest'
        )
        self.glows_hi90_file_by_repoint = {r["repointing"]: r["file_path"] for r in sp_hi90_query_result}

        self.hi_l2_query_result = imap_data_access.query(instrument='hi', data_level='l2', version='latest')

        hi_l3_query_result = imap_data_access.query(instrument='hi', data_level='l3', version='latest')
        self.existing_l3_maps = {(qr["descriptor"], qr["start_date"]): qr["file_path"] for qr in hi_l3_query_result}

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

    def get_maps_that_can_be_produced(self, descriptor: str) -> list[PossibleMapToProduce]:
        l2_descriptor = descriptor.replace('-sp-', '-nsp-')

        if 'h45' in descriptor:
            glows_file_by_repointing = self.glows_hi45_file_by_repoint
        elif 'h90' in descriptor:
            glows_file_by_repointing = self.glows_hi90_file_by_repoint
        else:
            raise ValueError("Expected map to be produced to use a single sensor!")

        l2_file_paths = [result['file_path'] for result in self.hi_l2_query_result if result["descriptor"] == l2_descriptor]
        possible_maps = []
        for l2_file_path in l2_file_paths:
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
                                         version='v001', descriptor=descriptor)

                possible_map_to_produce = PossibleMapToProduce(
                    input_files=set([l2_file_path] + glows_files),
                    input_metadata=input_metadata
                )
                possible_maps.append(possible_map_to_produce)
        return possible_maps
