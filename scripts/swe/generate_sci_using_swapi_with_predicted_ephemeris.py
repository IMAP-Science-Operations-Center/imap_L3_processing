import subprocess
import sys
from datetime import datetime, timedelta

import imap_data_access
from imap_data_access import DependencyFilePath, ProcessingInputCollection
from imap_data_access.processing_input import generate_imap_input

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

    input_files = [
        *spice_kernel_files,
        l2_file_name,
        swe_l1b_file_name,
        mag_l1d_file_name,
        swapi_l3a_file_name,
        swe_ancillary_file_name
    ]

    input_collection = ProcessingInputCollection(*[generate_imap_input(i) for i in input_files])
    input_collection.download_all_files()

    dependency_file_name = f'imap_swe_l3_sci_{day_as_string}_v002.json'
    output_filepath = DependencyFilePath(dependency_file_name)

    fullpath = output_filepath.construct_path()
    fullpath.parent.mkdir(parents=True, exist_ok=True)

    fullpath.write_text(input_collection.serialize())

    args_string = (f"imap_l3_data_processor.py "
                   f"--instrument swe --data-level l3 --descriptor sci "
                   f"--start-date {day_as_string} --version v004 "
                   f"--dependency {dependency_file_name}")

    return subprocess.Popen([sys.executable, *args_string.split(" ")])


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
