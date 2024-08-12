import uuid

import imap_data_access
from swapi.swapi_l3a_sw_proton_speed import main as calculate_sw_proton_speed
from spacepy.pycdf import CDF

file_paths = [result['file_path'] for result in imap_data_access.query(instrument='swapi', data_level='l2', descriptor="fake-menlo-5-sweeps")]
l2_file_path = imap_data_access.download(file_paths.pop())
l3_data = calculate_sw_proton_speed(str(l2_file_path))
l3_cdf_file_name = f'imap_swapi_l3a_fake-menlo-{uuid.uuid4()}_20240812_v001.cdf'
print(f"l3_cdf_file_name {l3_cdf_file_name}")
l3_cdf_file_path = f'test_data/{l3_cdf_file_name}'
l3_cdf = CDF(l3_cdf_file_path, '')
l3_cdf["epoch"] = l3_data.epoch
l3_cdf["proton_sw_speed"] = l3_data.proton_sw_speed
l3_cdf.close()
imap_data_access.upload(l3_cdf_file_path)
