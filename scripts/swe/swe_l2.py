import glob
import os
import math

import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.cdf.utils import load_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l2.swe_l2 import swe_l2

import imap_processing.tests.conftest
import imap_processing.swe.l1b.swe_l1b_science


# Don't have a cal file format for SWE yet, monkey-patch the lookup function
def fake_cal():
    return pd.DataFrame(
        {
            # probably want this to be 0 to infinity
            "met_time": [0, math.inf],
            "cem1": [1, 1],
            "cem2": [1, 1],
            "cem3": [1, 1],
            "cem4": [1, 1],
            "cem5": [1, 1],
            "cem6": [1, 1],
            "cem7": [1, 1],
        }
    )


imap_processing.swe.l1b.swe_l1b_science.read_in_flight_cal_data = fake_cal

use_l0_data = False
# Fake the spin data
if use_l0_data:
    data_start_time = 4.53051293e+08
else:
    data_start_time = 488937480
data_end_time = data_start_time + 60 * 1500
spin_df = imap_processing.tests.conftest.generate_spin_data.__wrapped__()(data_start_time, end_met=data_end_time)
spin_csv_file_path = "spin_data.spin.csv"
spin_df.to_csv(spin_csv_file_path, index=False)
os.environ["SPIN_DATA_FILEPATH"] = str(spin_csv_file_path)

# Making from l0
if use_l0_data:
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")
    l1b_dataset = swe_l1b(l1a_datasets[0], "002")
    l1b_path = write_cdf(l1b_dataset[0])
    loaded_l1b_dataset = load_cdf(l1b_path)
else:
    loaded_l1b_dataset = load_cdf("imap_swe_l1b_sci_20250630_v003.cdf")  # .isel(epoch=slice(None, 20))
l2_dataset = swe_l2(loaded_l1b_dataset, "002")
path = write_cdf(l2_dataset, compression=7)

print("wrote cdf to ", path)
