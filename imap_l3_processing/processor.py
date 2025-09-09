import abc
from pathlib import Path

import spiceypy
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import get_spice_parent_file_names


class Processor(abc.ABC):
    def __init__(self, dependencies: ProcessingInputCollection,
                 input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies

    @abc.abstractmethod
    def process(self) -> list[Path]:
        raise NotImplementedError()

    def get_parent_file_names(self, file_paths: list[Path | str] = None) -> list[str]:
        if file_paths:
            parent_file_names = [Path(parent_file_name).name for parent_file_name in file_paths]
        else:
            parent_file_names = [parent_file_name.name for parent_file_name in self.dependencies.get_file_paths()]

        return parent_file_names + get_spice_parent_file_names()


