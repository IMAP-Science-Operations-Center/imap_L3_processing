import argparse
import re
from datetime import datetime
from pathlib import Path

from spacepy.pycdf import CDF

from tests.test_helpers import get_test_data_path

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--l1c_folder_in_test_data")
parser.add_argument("-t", "--template")

args = parser.parse_args()

fake_l1c_folder = Path(args.template) if args.l1c_folder_in_test_data is not None else get_test_data_path(
    "hi/fake_l1c")
l1c_files = [filepath.name for filepath in fake_l1c_folder.iterdir()]

dates = []
for name in l1c_files:
    date_text = re.search(r"_(\d{8})_v", name)
    dates.append(datetime.strptime(date_text.group(1), "%Y%m%d"))

start_date = min(dates)
end_date = max(dates)

template_map_cdf = Path(args.template) if args.template is not None else get_test_data_path(
    "hi/IMAP_HI_90_maps_20000101_v02.cdf")

filename = f"imap_hi_45sensor-spacecraft-map_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_v001.cdf"
file_path = get_test_data_path("hi/fake_l2_maps") / filename

with CDF(str(file_path), masterpath=str(template_map_cdf)) as cdf:
    cdf.attrs["parents"] = ",".join(l1c_files)
    cdf["epoch"] = [start_date + (end_date - start_date) / 2]
