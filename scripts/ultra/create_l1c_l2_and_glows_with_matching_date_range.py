import shutil
from datetime import datetime, timedelta

import imap_data_access

from scripts.ultra.create_example_ultra_l1c_pset import _write_ultra_l1c_cdf_with_parents
from scripts.ultra.create_example_ultra_l2_map_with_power_law import _create_example_ultra_l2_map_with_power_law
from scripts.ultra.create_example_ultra_l3_survival_probabilities_cdf import create_survival_probabilities_file
from tests.test_helpers import get_test_instrument_team_data_path, get_run_local_data_path


def create_l1c_and_glows_with_matching_date_range(start_date: datetime, end_date: datetime):
    process_date = start_date
    repoint_counter = 0
    l1c_filepaths = []
    all_paths = []

    base_path = f'ultra/{start_date:%Y%m%d}-{end_date:%Y%m%d}'
    shutil.rmtree(get_run_local_data_path(base_path), ignore_errors=True)

    version_number = 10

    while process_date < end_date:
        repoint_counter += 1

        l1c_file_path = get_run_local_data_path(
            f'{base_path}/ultra_l1c/imap_ultra_l1c_45sensor-spacecraftpset_{process_date:%Y%m%d}-repoint{repoint_counter:05}_v{version_number:03}.cdf'
        )
        l1c_file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_ultra_l1c_cdf_with_parents(date=process_date.isoformat(), out_path=l1c_file_path)
        l1c_filepaths.append(str(l1c_file_path))
        all_paths.append(str(l1c_file_path))

        glows_l3e_file_path = get_run_local_data_path(
            f'{base_path}/glows_l3e/imap_glows_l3e_survival-probability-ultra_{process_date:%Y%m%d}_v{version_number:03}.cdf'
        )
        glows_l3e_file_path.parent.mkdir(parents=True, exist_ok=True)

        glows_input_path = get_test_instrument_team_data_path('glows/probSur.Imap.Ul.V0_2009.000.dat')
        create_survival_probabilities_file(glows_file_path=glows_input_path, date_for_file=process_date,
                                           cdf_file_path=get_run_local_data_path(glows_l3e_file_path))
        all_paths.append(str(glows_l3e_file_path))
        process_date = process_date + timedelta(days=1)

    l2_out_path = get_run_local_data_path(
        f'{base_path}/imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-6mo_{start_date:%Y%m%d}_v{version_number:03}.cdf'
    )
    l2_out_path.parent.mkdir(parents=True, exist_ok=True)

    create_example_ultra_l2_with_parents(l2_out_path, l1c_filepaths)
    all_paths.append(str(l2_out_path))

    return all_paths


def create_example_ultra_l2_with_parents(out_path, parents: list[str]):
    _create_example_ultra_l2_map_with_power_law(out_path=out_path, parents=parents)


if __name__ == "__main__":
    paths = create_l1c_and_glows_with_matching_date_range(datetime(2025, 4, 15, 12), datetime(2025, 4, 19, 12))
    for path in paths:
        imap_data_access.upload(path)
