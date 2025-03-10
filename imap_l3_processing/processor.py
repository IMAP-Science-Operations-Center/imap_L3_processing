from typing import List

from imap_l3_processing.models import UpstreamDataDependency, InputMetadata


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency], input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies
