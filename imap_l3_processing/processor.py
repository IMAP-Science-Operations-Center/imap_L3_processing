from pathlib import Path

import spiceypy
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.models import InputMetadata


class Processor:
    def __init__(self, dependencies: ProcessingInputCollection,
                 input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies

    def get_parent_file_names(self, file_paths: list[Path] = None) -> list[str]:
        if file_paths:
            parent_file_names = [parent_file_name.name for parent_file_name in file_paths]
        else:
            parent_file_names = [parent_file_name.name for parent_file_name in self.dependencies.get_file_paths()]

        count = spiceypy.ktotal('ALL')
        for i in range(0, count):
            file = Path(spiceypy.kdata(i, 'ALL')[0]).name
            parent_file_names.append(file)

        return parent_file_names
