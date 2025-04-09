from imap_l3_processing.models import UpstreamDataDependency, InputMetadata
from imap_l3_processing.processor import Processor


class CodiceProcessor(Processor):
    def __init__(self, dependencies: list[UpstreamDataDependency], input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        pass
