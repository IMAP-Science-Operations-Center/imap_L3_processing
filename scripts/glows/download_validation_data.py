import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["IMAP_DATA_DIR"] = str(Path(__file__).parent.parent.parent / "glows_l3_validation_data")

import imap_data_access

ancillary_descriptors = [
    ("calibration-data", "20100101"),
    ("map-of-extra-helio-bckgrd", "20100101"),
    ("pipeline-settings", "20100101"),
    ("time-dep-bckgrd", "20100101"),

    ("uv-anisotropy-1CR", "20100101"),
    ("WawHelioIonMP", "20100101"),
    ("bad-days-list", "20100101"),
    ("pipeline-settings-l3bcde", "20100101"),

    ("plasma-speed-2010a", "20100101"),
    ("proton-density-2010a", "20100101"),
    ("uv-anisotropy-2010a", "20100101"),
    ("photoion-2010a", "20100101"),
    ("lya-2010a", "20100101"),
    ("electron-density-2010a", "20100101"),

    ("lya", "19470303"),
    ("e-dens", "19470303"),
    ("p-dens", "19470303"),
    ("uv-anis", "19470303"),
    ("phion", "19470303"),
    ("speed", "19470303"),
]

for descriptor, start_date in ancillary_descriptors:
    [result] = imap_data_access.query(
        table="ancillary",
        instrument="glows",
        descriptor=descriptor,
        start_date=start_date,
        end_date=(datetime.strptime(start_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"),
        version="latest"
    )

    print(imap_data_access.download(result["file_path"]))

data_files = [
    ("l2", "hist"),
    ("l3a", "hist"),
    ("l3b", "ion-rate-profile"),
    ("l3c", "sw-profile"),
    ("l3d", "solar-hist")
]

for data_level, descriptor in data_files:
    for qr in imap_data_access.query(instrument="glows", data_level=data_level, descriptor=descriptor, version="latest"):
        print(imap_data_access.download(qr["file_path"]))

# download l3bc archive
for qr in imap_data_access.query(table="ancillary", instrument="glows", descriptor="l3b-archive", version="latest"):
    print(imap_data_access.download(qr["file_path"]))
