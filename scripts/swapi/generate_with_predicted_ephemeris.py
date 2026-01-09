import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import imap_data_access
from imap_data_access import DependencyFilePath, ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input

from imap_l3_processing.utils import get_spice_kernels_file_names, SpiceKernelTypes

SPICE_KERNELS = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.IMAPFrames,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.EphemerisReconstructed,
    SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.EphemerisPredicted,
    SpiceKernelTypes.PlanetaryEphemeris
]


def generate_swapi_for_given_day(descriptor: str, day: datetime):
    spice_kernel_files = get_spice_kernels_file_names(day, day + timedelta(days=1), SPICE_KERNELS)
    day_as_string = day.strftime('%Y%m%d')
    l2_file_path = imap_data_access.query(instrument='swapi', data_level='l2', start_date=day_as_string,
                                          end_date=day_as_string, version='latest')[0]['file_path']

    template_json = Path('scripts/swapi/imap_swapi_l3a_proton-sw_dependency_template.json')

    input_files = [
        *spice_kernel_files,
        Path(l2_file_path).name
    ]

    input_collection = ProcessingInputCollection(*[generate_imap_input(i) for i in input_files])
    input_collection.deserialize(template_json.read_text())

    input_collection.download_all_files()

    dependency_file_name = f'imap_swapi_l3a_{descriptor}_{day_as_string}_v002.json'
    output_filepath = DependencyFilePath(dependency_file_name)

    fullpath = output_filepath.construct_path()
    fullpath.parent.mkdir(parents=True, exist_ok=True)

    fullpath.write_text(input_collection.serialize())

    args_string = (f"imap_l3_data_processor.py "
                   f"--instrument swapi --data-level l3a --descriptor {descriptor} "
                   f"--start-date {day_as_string} --version v002 "
                   f"--dependency {dependency_file_name}")

    return subprocess.Popen([sys.executable, *args_string.split(" ")])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dates = [datetime.strptime(arg, "%Y%m%d") for arg in sys.argv[1:]]
    else:
        dates = [datetime(2025, 12, 15),
                 datetime(2025, 12, 16),
                 datetime(2025, 12, 17),
                 datetime(2025, 12, 18)]
    pids = [generate_swapi_for_given_day('proton-sw', date) for date in dates]

    for pid in pids:
        pid.wait()
