import uuid

import imap_data_access
from swapi.swapi_l3a_sw_proton_speed import main as calculate_sw_proton_speed
from spacepy.pycdf import CDF
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instrument")
parser.add_argument("--level")
parser.add_argument("--start_date")
parser.add_argument("--end_date")
parser.add_argument("--version")
parser.add_argument("--dependency")

args = parser.parse_args()
dependencies = json.loads(args.dependency.replace("'", '"'))

file_paths = [result['file_path'] for result in imap_data_access.query(instrument=dependencies[0]["instrument"], data_level=dependencies[0]["data_level"], descriptor=dependencies[0]["descriptor"])]
l2_file_path = imap_data_access.download(file_paths.pop())
l3_data = calculate_sw_proton_speed(str(l2_file_path))
l3_cdf_file_name = f'imap_{args.instrument}_{args.level}_fake-menlo-{uuid.uuid4()}_{args.start_date}_{args.version}.cdf'
print(f"l3_cdf_file_name {l3_cdf_file_name}")
l3_cdf_file_path = f'test_data/{l3_cdf_file_name}'
l3_cdf = CDF(l3_cdf_file_path, '')
l3_cdf["epoch"] = l3_data.epoch
l3_cdf["proton_sw_speed"] = l3_data.proton_sw_speed
l3_cdf.close()
imap_data_access.upload(l3_cdf_file_path)
