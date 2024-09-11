import imap_data_access

from imap_processing.processor import Processor


class HITL3Processor(Processor):
    def process(self):
        l2_data_dependency = self.dependencies[0]
        imap_data_access.query(instrument=l2_data_dependency.instrument, data_level=l2_data_dependency.data_level,
                               descriptor=l2_data_dependency.descriptor, start_date=self.format_time(self.start_date),
                               end_date=self.format_time(self.end_date), version="latest")
        l2_mag_dependency = self.dependencies[1]
        imap_data_access.query(instrument=l2_mag_dependency.instrument, data_level=l2_mag_dependency.data_level,
                               descriptor=l2_mag_dependency.descriptor, start_date=self.format_time(self.start_date),
                               end_date=self.format_time(self.end_date), version="latest")
