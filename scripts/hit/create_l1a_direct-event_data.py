import os
from datetime import timedelta, datetime
from pathlib import Path

import numpy as np
from bitstring import BitStream
from spacepy.pycdf import CDF

path = Path(__file__)
binary_data_path = path.parent.parent.parent / "tests" / "test_data" / "hit" / "pha_events" / "pha_binary"

start_time = datetime(year=2010, month=1, day=6)
time_delta = timedelta(minutes=10)

epoch = []
pha_raw = []

for i, filename in enumerate(os.listdir(str(binary_data_path))):
    bitstream = BitStream(filename=str(binary_data_path / filename))
    epoch.append(start_time + i * time_delta)

    binary = bitstream.bin
    print(len(binary))
    pha_raw.append(binary)

with CDF("tests/test_data/hit/pha_events/fake-menlo-imap_hit_l1a_pulse-height-events_20100106_v004",
         masterpath='') as cdf:
    cdf.new("epoch", np.array(epoch))
    cdf.new("pha_raw", pha_raw)
