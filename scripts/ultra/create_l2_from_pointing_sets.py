import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
from imap_data_access import ProcessingInputCollection
from imap_data_access.file_validation import generate_imap_file_path
from imap_data_access.processing_input import generate_imap_input

output_dir = r"C:\Users\Harrison\Development\imap_L3_processing\generate_ultra_l2"


def create_input_file(sensor: str, start: datetime, end: datetime) -> str:
    static_files = [
        "imap_recon_20250415_20260415_v01.bsp",
        "imap_science_105.tf",
        "imap_008.tf",
        "naif016.tls",
        "de440.bsp",
        "imap_sclk_005.tsc",
        "imap_2025_105_2026_105_01.ah.bc",
        "imap_dps_2025_105_2026_105_008.ah.bc",
    ]

    inputs = ProcessingInputCollection()

    inputs.add([generate_imap_input(f) for f in static_files])

    start_repointings = datetime(2025, 9, 29)

    date = start
    while date < end:
        repointing = int((date - start_repointings) / timedelta(days=1))
        inputs.add(generate_imap_input(
            f"imap_ultra_l1c_{sensor}sensor-heliopset_{date.strftime('%Y%m%d')}-repoint{repointing:05}_v000.cdf"))
        date += timedelta(days=1)
    dependency_file = f"imap_ultra_l2_{sensor}sensor-de-{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}_{start.strftime('%Y%m%d')}_v001.json"
    dependency_path = generate_imap_file_path(dependency_file).construct_path()
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_text(inputs.serialize())
    return dependency_file


def generate_map(descriptor: str, start: datetime, end: datetime):
    sensor = descriptor.split('-')[0][1:]
    dependency_file = create_input_file(sensor, start, end)

    env = {
        **os.environ,
        "IMAP_DATA_ACCESS_URL": "https://api.dev.imap-mission.com",
        "IMAP_DATA_DIR": output_dir,
    }

    version = "v000"
    formatted_start_date = start_date.strftime('%Y%m%d')
    output_filename = f"imap_ultra_l2_{descriptor}_{formatted_start_date}_{version}.cdf"
    command = f"imap_cli --instrument ultra --data-level l2 --descriptor {descriptor} --start-date {formatted_start_date} --version {version} --dependency {dependency_file}"
    print(command)
    if not generate_imap_file_path(output_filename).construct_path().exists():
        subprocess.run(command.split(" "), env=env)
    else:
        print(f"Skipping generation of {output_filename}, it already exists")


if __name__ == "__main__":
    imap_data_access.config["DATA_DIR"] = Path(output_dir)

    start_date = datetime(2025, 4, 16)
    end_date = datetime(2025, 10, 15)

    duration = '6mo'
    descriptor = f"u90-ena-h-hf-nsp-full-hae-2deg-6mo"
    generate_map(descriptor, start_date, end_date)
    # for sensor in [45, 90]:
    #     for side in ['ram', 'anti', 'full']:
    #         for pixelation in [4, 6]:
    #             for start_date, end_date in [
    #                 (datetime(2025, 4, 15), datetime(2025, 10, 15)),
    #                 (datetime(2025, 10, 15), datetime(2026, 4, 15))
    #             ]:
    #                 descriptor = f"u{sensor}-ena-h-hf-nsp-{side}-hae-{pixelation}deg-{duration}"
    #                 generate_map(descriptor, start_date, end_date)
