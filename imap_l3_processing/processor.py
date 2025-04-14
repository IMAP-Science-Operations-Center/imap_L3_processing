from typing import List

from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.models import UpstreamDataDependency, InputMetadata


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency] | ProcessingInputCollection,
                 input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies
