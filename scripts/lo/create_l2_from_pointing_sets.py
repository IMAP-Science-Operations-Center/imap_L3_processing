import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
from imap_data_access import ProcessingInputCollection
from imap_data_access.file_validation import generate_imap_file_path
from imap_data_access.processing_input import generate_imap_input

input_dir = r"C:\Users\Harrison\Development\imap_L3_processing\scripts\lo\data"
output_dir = r"C:\Users\Harrison\Development\imap_L3_processing\generate_lo_l2\data"


def create_input_file() -> str:
    static_files = [
        "imap_recon_20250415_20260415_v01.bsp",
        "imap_science_110.tf",
        "imap_130.tf",
        "naif020.tls",
        "de440.bsp",
        "imap_sclk_005.tsc",
        "imap_2025_105_2026_105_01.ah.bc",
        "imap_dps_2025_105_2026_105_009.ah.bc",
        "imap_lo_esa-eta-fit-factors_20240101_v001.csv",
        "imap_lo_l1c_pset_20250415-repoint01000_v006.cdf",
        "imap_lo_l1c_pset_20250420-repoint01005_v002.cdf",
        "imap_lo_l1c_pset_20250424-repoint01009_v002.cdf",
        "imap_lo_l1c_pset_20250429-repoint01014_v002.cdf",
        "imap_lo_l1c_pset_20250504-repoint01019_v002.cdf"
    ]

    inputs = ProcessingInputCollection()

    inputs.add([generate_imap_input(f) for f in static_files])

    dependency_file = f"imap_lo_l2_90sensor-de-20250415-20250504_20250415_v001.json"
    dependency_path = generate_imap_file_path(dependency_file).construct_path()
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_text(inputs.serialize())
    return dependency_file


def generate_map(descriptor: str, start: datetime):
    dependency_file = create_input_file()

    env = {
        **os.environ,
        "IMAP_DATA_ACCESS_URL": "https://api.dev.imap-mission.com",
        "IMAP_DATA_DIR": output_dir,
    }

    version = "v000"
    formatted_start_date = start.strftime('%Y%m%d')
    output_filename = f"imap_lo_l2_{descriptor}_{formatted_start_date}_{version}.cdf"
    command = f"imap_cli --instrument lo --data-level l2 --descriptor {descriptor} --start-date {formatted_start_date} --version {version} --dependency {dependency_file}"
    print(command)
    if not generate_imap_file_path(output_filename).construct_path().exists():
        subprocess.run(command.split(" "), env=env)
    else:
        print(f"Skipping generation of {output_filename}, it already exists")


if __name__ == "__main__":
    imap_data_access.config["DATA_DIR"] = Path(output_dir)

    for file in Path(input_dir).iterdir():
        p = generate_imap_file_path(file.name).construct_path()
        p.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file, p)

    start_date = datetime(2025, 4, 15)
    descriptors = [
        f"l090-ena-h-hf-nsp-ram-hae-6deg-1yr",
        f"l090-ena-h-sf-nsp-ram-hae-6deg-1yr",
        f"l090-enanbs-h-hf-nsp-ram-hae-6deg-1yr",
        f"l090-enanbs-h-sf-nsp-ram-hae-6deg-1yr",
    ]
    for descriptor in descriptors:
        generate_map(descriptor, start_date)