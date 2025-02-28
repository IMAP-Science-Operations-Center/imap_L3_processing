from datetime import datetime

import numpy as np
from bitstring import BitStream
from spacepy.pycdf import CDF

bitstream = BitStream(filename="tests/test_data/hit/pha_events/full_event_record_buffer.bin")

with CDF("tests/test_data/hit/pha_events/fake-menlo-imap_hit_l1a_pulse-height-events_20100106_v003", '') as cdf:
    cdf["epoch"] = np.array([datetime(year=2010, month=1, day=6)])
    cdf["pha_raw"] = np.array([bitstream.bin])
