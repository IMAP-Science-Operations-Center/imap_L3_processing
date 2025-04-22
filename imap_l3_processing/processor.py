from typing import List

from imap_data_access.processing_input import ProcessingInputCollection
from spiceypy import spiceypy

from imap_l3_processing.models import UpstreamDataDependency, InputMetadata


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency] | ProcessingInputCollection,
                 input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies

    def get_parent_file_names(self):
        parent_file_names = [parent_file_name.name for parent_file_name in self.dependencies.get_file_paths()]

        count = spiceypy.ktotal('ALL')
        for i in range(0, count):
            file = spiceypy.kdata(i, 'ALL')[0]
            parent_file_names.append(file)

        return parent_file_names
