from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.direct_event.codice_l3_dependencies import CodiceL3Dependencies
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.processor import Processor


class CodiceProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            dependencies = CodiceL3Dependencies.fetch_dependencies(self.dependencies)
            self.process_l3a(dependencies)

    def process_l3a(self, dependencies: CodiceL3Dependencies):
        pass
