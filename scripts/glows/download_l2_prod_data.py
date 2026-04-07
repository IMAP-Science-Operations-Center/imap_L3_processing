
from datetime import datetime
from pathlib import Path

import dotenv
import imap_data_access
import requests

from imap_l3_processing.utils import SpiceKernelTypes

dotenv.load_dotenv(".env.prod")

from tests.test_helpers import get_run_local_data_path

GLOWS_L3E_REQUIRED_SPICE_KERNELS: list[SpiceKernelTypes] = [
    SpiceKernelTypes.ScienceFrames, SpiceKernelTypes.EphemerisReconstructed, SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude, SpiceKernelTypes.PlanetaryEphemeris, SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.SpacecraftClock
]


def datetime_to_j2000_seconds(date: datetime) -> int:
    return int((date - datetime(2000, 1, 1, 12)).total_seconds())


glows_prod_data_path = get_run_local_data_path("glows_prod_data")
imap_data_access.config["DATA_DIR"] = glows_prod_data_path

l2_query_response = imap_data_access.query(instrument="glows", data_level="l2", version="latest")

for file in l2_query_response:
    print(f"downloading: {Path(file['file_path']).name}")
    imap_data_access.download(file["file_path"])

start_of_l2 = datetime(2025, 11, 28)
end_of_l2 = datetime(2026, 3, 15)

metakernel_params = {
    "file_types": [t.value for t in GLOWS_L3E_REQUIRED_SPICE_KERNELS],
    "start_time": datetime_to_j2000_seconds(start_of_l2),
    "end_time": datetime_to_j2000_seconds(end_of_l2),
    "list_files": "true"
}
resp = requests.get("https://api.imap-mission.com/metakernel", params=metakernel_params)
for spice_file in resp.json():
    print(f"downloading: {spice_file}")
    imap_data_access.download(spice_file)
