import imap_data_access

from imap_processing.processor import Processor
from imap_processing.utils import format_time


class HITL3Processor(Processor):
    def process(self):
        l2_data_dependency = self.dependencies[0]
        imap_data_access.query(instrument=l2_data_dependency.instrument, data_level=l2_data_dependency.data_level,
                               descriptor=l2_data_dependency.descriptor,
                               start_date=format_time(self.input_metadata.start_date),
                               end_date=format_time(self.input_metadata.end_date), version="latest")
        l2_mag_dependency = self.dependencies[1]
        imap_data_access.query(instrument=l2_mag_dependency.instrument, data_level=l2_mag_dependency.data_level,
                               descriptor=l2_mag_dependency.descriptor,
                               start_date=format_time(self.input_metadata.start_date),
                               end_date=format_time(self.input_metadata.end_date), version="latest")
