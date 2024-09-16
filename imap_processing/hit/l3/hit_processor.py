import imap_data_access
from spacepy.pycdf import CDF

from imap_processing import utils
from imap_processing.hit.l3.utils import read_l2_hit_data
from imap_processing.processor import Processor
from imap_processing.utils import format_time, read_l2_mag_data

HIT_L2_DESCRIPTOR = "let1-rates3600-fake-menlo"
MAG_L2_DESCRIPTOR = "fake-menlo-mag-SC-1min"


class HITL3Processor(Processor):
    def process(self):

        try:
            data_dependency = next(
                dependency for dependency in self.dependencies if dependency.descriptor == HIT_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {HIT_L2_DESCRIPTOR} dependency.")

        try:
            mag_dependency = next(
                dependency for dependency in self.dependencies if dependency.descriptor == MAG_L2_DESCRIPTOR)
        except StopIteration:
            raise ValueError(f"Missing {MAG_L2_DESCRIPTOR} dependency.")

        l2_data_dependency_path = utils.download_dependency(data_dependency)
        l2_mag_dependency_path = utils.download_dependency(mag_dependency)

        hit_data = read_l2_hit_data(CDF(str(l2_data_dependency_path)))
        mag_data = read_l2_mag_data(CDF(str(l2_mag_dependency_path)))
