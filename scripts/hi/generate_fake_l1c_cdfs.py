from datetime import datetime, timedelta

from spacepy import pycdf
from spacepy.pycdf import CDF

from tests.test_helpers import get_test_data_path

num_fake_files = 90

for i in range(num_fake_files):
    date_to_set = datetime(month=4, year=2025, day=16, hour=12) + timedelta(days=i)
    logical_source = f"imap_hi_l1c_45sensor-pset_{date_to_set.strftime('%Y%m%d')}_v001"
    output_filename = str(get_test_data_path(f"hi/fake_l1c/{logical_source}.cdf"))
    with CDF(output_filename,
             masterpath=str(get_test_data_path("hi/imap_hi_l1c_45sensor-pset_20250415_v001.cdf"))) as cdf:
        del cdf['epoch']
        cdf.new("epoch", [date_to_set], type=pycdf.const.CDF_TIME_TT2000)
