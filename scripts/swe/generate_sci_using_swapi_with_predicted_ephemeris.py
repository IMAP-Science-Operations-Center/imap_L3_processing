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
    SpiceKernelTypes.SpacecraftClock,
    SpiceKernelTypes.EphemerisPredicted,
    SpiceKernelTypes.PlanetaryEphemeris,
    SpiceKernelTypes.PlanetaryConstants,
    SpiceKernelTypes.PointingAttitude,
]


def generate_swe_for_given_day(day: datetime):
    spice_kernel_files = get_spice_kernels_file_names(day, day + timedelta(days=1), SPICE_KERNELS)
    day_as_string = day.strftime('%Y%m%d')
    l2_file_name = imap_data_access.query(instrument='swe', data_level='l2', start_date=day_as_string,
                                          end_date=day_as_string, version='latest')[0]['file_path']

    mag_l1d_file_name = imap_data_access.query(instrument='mag', data_level='l1d', start_date=day_as_string,
                                               end_date=day_as_string, version='latest', descriptor='norm-dsrf')[0][
        'file_path']

    swapi_l3a_file_name = f'imap_swapi_l3a_proton-sw_{day_as_string}_v002.cdf'

    swe_l1b_file_name = imap_data_access.query(instrument='swe', data_level='l1b', start_date=day_as_string,
                                               end_date=day_as_string, version='latest', descriptor='sci')[0][
        'file_path']

    swe_ancillary_file_name = "imap_swe_config_20251119_v002.json"

    swe_data_as_json = []

    swe_data_as_json.append(
        {
            "type": "spice",
            "files": spice_kernel_files
        }
    )
    swe_data_as_json.append(
        {
            "type": "science",
            "files": [l2_file_name]
        }
    )
    swe_data_as_json.append(
        {
            "type": "science",
            "files": [swe_l1b_file_name]
        }
    )
    swe_data_as_json.append(
        {
            "type": "science",
            "files": [mag_l1d_file_name]
        }
    )
    swe_data_as_json.append(
        {
            "type": "science",
            "files": [swapi_l3a_file_name]
        }
    )
    swe_data_as_json.append(
        {
            "type": "ancillary",
            "files": [swe_ancillary_file_name]
        }
    )
    swe_data_as_json.append(
        {
            "type": "spice",
            "files": spice_kernel_files
        }
    )

    dependency_file_name = f'imap_swe_l3_sci_{day_as_string}_v002.json'
    output_filepath = DependencyFilePath(dependency_file_name)

    fullpath = output_filepath.construct_path()
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fullpath, mode='wt') as output_file:
        json.dump(swe_data_as_json, output_file, indent=4)

    args_string = (f"imap_l3_data_processor.py "
                   f"--instrument swe --data-level l3 --descriptor sci "
                   f"--start-date {day_as_string} --version v004 "
                   f"--dependency {dependency_file_name}")

    pid = subprocess.Popen([sys.executable, *args_string.split(" ")])

    return pid

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dates = [datetime.strptime(arg, "%Y%m%d") for arg in sys.argv[1:]]
    else:
        dates = [
            datetime(2025, 12, 15),
            datetime(2025, 12, 16),
            datetime(2025, 12, 17),
            datetime(2025, 12, 18),
        ]

    pids = [generate_swe_for_given_day(date) for date in dates]

    for pid in pids:
        pid.wait()
