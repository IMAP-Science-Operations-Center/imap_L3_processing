import json
import subprocess
import sys
from datetime import datetime, timedelta

import imap_data_access
from imap_data_access import DependencyFilePath

from imap_l3_processing.utils import get_spice_kernels_file_names, SpiceKernelTypes

SPICE_KERNELS = [
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.IMAPFrames,
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.EphemerisReconstructed,
    SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.EphemerisPredicted
]


def generate_swapi_for_given_day(day: datetime):
    spice_kernel_files = get_spice_kernels_file_names(day, day + timedelta(days=1), SPICE_KERNELS)
    day_as_string = day.strftime('%Y%m%d')
    l2_file_name = imap_data_access.query(instrument='swapi', data_level='l2', start_date=day_as_string,
                                          end_date=day_as_string, version='latest')[0]['file_path']
    with open('scripts/swapi/imap_swapi_l3a_proton-sw_dependency_template.json') as dependency_template_file:
        dependency_template = json.load(dependency_template_file)
    dependency_template[0]['files'] = spice_kernel_files
    dependency_template[1]['files'] = [l2_file_name]

    dependency_file_name = f'imap_swapi_l3a_proton-sw_{day_as_string}_v002.json'
    output_filepath = DependencyFilePath(dependency_file_name)

    fullpath = output_filepath.construct_path()
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fullpath, mode='wt') as output_file:
        json.dump(dependency_template, output_file, indent=4)

    args_string = (f"imap_l3_data_processor.py "
                   f"--instrument swapi --data-level l3a --descriptor proton-sw "
                   f"--start-date {day_as_string} --version v002 "
                   f"--dependency {dependency_file_name}")

    subprocess.run([sys.executable, *args_string.split(" ")])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dates = [datetime.strptime(arg, "%Y%m%d") for arg in sys.argv[1:]]
    else:
        dates = [datetime(2025, 12, 15)]
    for date in dates:
        generate_swapi_for_given_day(date)
