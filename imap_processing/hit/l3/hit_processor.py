import imap_data_access
from spacepy.pycdf import CDF
from spiceypy import spiceypy

from imap_processing import utils
from imap_processing.hit.l3.utils import read_l2_hit_data, calculate_unit_vector
from imap_processing.processor import Processor
from imap_processing.utils import format_time, read_l1d_mag_data


class HitProcessor(Processor):
    def process(self):
        pass
