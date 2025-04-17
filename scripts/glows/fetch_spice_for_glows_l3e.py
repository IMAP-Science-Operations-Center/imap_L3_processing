from datetime import datetime, timedelta

from imap_l3_processing import spice_wrapper
from imap_l3_processing.glows.l3e.glows_l3e_utils import determine_call_args_for_l3e_executable

spice_wrapper.furnish()


def make_input_args(start_time, number_of_points, elongation):
    lines = []
    for i in range(number_of_points):
        t = start_time + timedelta(days=i)
        lines.append(determine_call_args_for_l3e_executable(t, elongation))
    return lines


for line in make_input_args(datetime(2025, 4, 15, 12)
        , 365, 90):
    output = f"./survProbHi {line}"
    output = output.replace(' 2025', ' 2015')
    output = output.replace(' 2026', ' 2016')
    print(output)
