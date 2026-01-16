from datetime import timedelta, datetime
from pathlib import Path

import imap_data_access
import requests
from imap_data_access import ProcessingInputCollection
from imap_data_access.file_validation import generate_imap_file_path
from imap_data_access.processing_input import generate_imap_input

from imap_l3_processing.utils import get_spice_kernels_file_names, SpiceKernelTypes

REFERENCE_FRAMES = ['sf', 'hf']
POINTING_SET_DESCRIPTORS_FROM_REF_FRAME = {
    'sf':'spacecraftpset',
    'hf':'heliopset'
}
SENSORS = ['u90', 'u45']
START_DATE = datetime(2025, 9, 29)
END_DATE = datetime(2025,12,29)

SENSOR_TO_SENSORNAME = {
    'u90':'90sensor',
    'u45':'45sensor'
}

SPICE_KERNELS = [
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.IMAPFrames,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.ScienceFrames
]

def create_l2_dependency_files(sensor, reference_frame):
    inputs = ProcessingInputCollection()
    pointing_set_descriptor = f'{SENSOR_TO_SENSORNAME[sensor]}-{POINTING_SET_DESCRIPTORS_FROM_REF_FRAME[reference_frame]}'
    dependency_file_descriptor = f'{sensor}-{reference_frame}'

    spice_files = []
    for kernel_type in SPICE_KERNELS:
        response = requests.get(
            imap_data_access.config["DATA_ACCESS_URL"] + f"/spice-query?type={kernel_type.value}&start_time=0",
        )
        response.raise_for_status()
        file_json = response.json()
        for spice_file in file_json:
            spice_start_date = datetime.strptime(spice_file["min_date_datetime"], "%Y-%m-%d, %H:%M:%S")
            spice_end_date = datetime.strptime(spice_file["max_date_datetime"], "%Y-%m-%d, %H:%M:%S")
            if spice_start_date <= END_DATE and START_DATE < spice_end_date:
                spice_files.append(Path(spice_file["file_name"]).name)
    query_results = imap_data_access.query(
        descriptor=pointing_set_descriptor,
        instrument='ultra',
        data_level='l1c',
        start_date=START_DATE.strftime('%Y%m%d'),
        end_date=END_DATE.strftime('%Y%m%d'),
        version='latest'
    )

    desired_repointings = [*range(21,28+1), 31, 32, *range(51,57+1), *range(62, 66+1)]

    pset_files = [Path(file['file_path']).name for file in query_results if file['repointing'] in desired_repointings]

    inputs.add([generate_imap_input(f) for f in spice_files+pset_files])

    dependency_file = f"imap_ultra_l2_{dependency_file_descriptor}_{START_DATE.strftime('%Y%m%d')}_v001.json"
    dependency_path = generate_imap_file_path(dependency_file).construct_path()
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_text(inputs.serialize())

    return dependency_file


if __name__ == '__main__':
    for sensor in SENSORS:
        for reference_frame in REFERENCE_FRAMES:
            create_l2_dependency_files(sensor, reference_frame)
