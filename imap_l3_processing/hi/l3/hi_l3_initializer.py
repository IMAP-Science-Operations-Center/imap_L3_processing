import logging
from dataclasses import dataclass
from datetime import datetime

import imap_data_access
from imap_data_access.file_validation import ImapFilePath
from imap_data_access.file_validation import ScienceFilePath
from spacepy.pycdf import CDF

from imap_l3_processing.models import InputMetadata

logger = logging.getLogger(__name__)


@dataclass
class PossibleMapToProduce:
    input_files: set[str]
    input_metadata: InputMetadata


class HiL3Initializer:
    @staticmethod
    def get_maps_that_should_be_produced(descriptor: str) -> list[PossibleMapToProduce]:
        possible_maps = HiL3Initializer.get_maps_that_can_be_produced(descriptor)
        l3_query_result = imap_data_access.query(instrument='hi', data_level='l3', descriptor=descriptor,
                                                 version='latest')
        already_made = {}
        for result in l3_query_result:
            already_made[result["start_date"]] = result

        maps_to_make = []
        for map in possible_maps:
            start_time = map.input_metadata.start_date.strftime("%Y%m%d")
            if l3_result := already_made.get(start_time):
                if map.input_files.issubset(get_parent_file_names(l3_result['file_path'])):
                    continue
                map.input_metadata.version = f'v{int(l3_result['version'][1:]) + 1:03}'
            maps_to_make.append(map)
        return maps_to_make

    @staticmethod
    def get_maps_that_can_be_produced(descriptor: str) -> list[PossibleMapToProduce]:
        l2_descriptor = descriptor.replace('-sp-', '-nsp-')

        if 'h45' in descriptor:
            sensor_angle = 45
        elif 'h90' in descriptor:
            sensor_angle = 90
        glows_descriptor = f'survival-probability-hi-{sensor_angle}'

        l2_query_result = imap_data_access.query(instrument='hi', data_level='l2', descriptor=l2_descriptor,
                                                 version='latest')
        glows_query_result = imap_data_access.query(instrument='glows', data_level='l3e', descriptor=glows_descriptor,
                                                    version='latest')
        glows_repointings = {result['repointing'] for result in glows_query_result}

        l2_file_paths = [result['file_path'] for result in l2_query_result]

        possible_maps = []

        for l2_file_path in l2_file_paths:
            start_date = datetime.strptime(ScienceFilePath(l2_file_path).start_date, "%Y%m%d")
            l1c_names = get_parent_file_names(l2_file_path)
            # ASSUMING l1c have repointing in file names
            l1c_repointings = set()
            for l1 in l1c_names:
                try:
                    l1c_repointings.add(ScienceFilePath(l1).repointing)
                except ImapFilePath.InvalidImapFileError as e:
                    continue

            if l1c_repointings.issubset(glows_repointings):
                relevant_glows = {
                    result['file_path']
                    for result in glows_query_result
                    if result['repointing'] in l1c_repointings
                }
                possible_maps.append(
                    PossibleMapToProduce(input_files=relevant_glows.union([l2_file_path]),
                                         input_metadata=InputMetadata(
                                             instrument='hi',
                                             data_level='l3',
                                             start_date=start_date,
                                             end_date=None,
                                             version='v001',
                                             descriptor=descriptor
                                         )
                                         ))
        return possible_maps


def get_parent_file_names(l2_file: str) -> set[str]:
    l2_path = imap_data_access.download(l2_file)
    with CDF(str(l2_path)) as cdf:
        parents = cdf.attrs.get("Parents")
        if parents is None:
            logger.info("Parents attribute was not found on the L2.")
            return set()
        return set(parents)
