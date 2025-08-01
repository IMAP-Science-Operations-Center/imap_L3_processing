import shutil
from datetime import datetime

import imap_data_access

from imap_l3_processing.utils import furnish_local_spice
from scripts.ultra.create_example_ultra_l2_map_with_power_law import _create_example_ultra_l2_map_with_power_law
from scripts.ultra.create_example_ultra_l3_survival_probabilities_cdf import create_survival_probabilities_file
from tests.test_helpers import get_test_instrument_team_data_path, get_run_local_data_path


def create_l1c_and_glows_with_matching_date_range(start_date: datetime, end_date: datetime):
    all_paths = []

    base_path = f'ultra/{start_date:%Y%m%d}-{end_date:%Y%m%d}'
    shutil.rmtree(get_run_local_data_path(base_path), ignore_errors=True)

    version_number = 11

    l1c_path = get_run_local_data_path('ultra/l1c_from_nat')

    l1c_filepaths = [
        l1c_path / 'imap_ultra_l1c_90sensor-spacecraftpset_20250515-repoint00001_v001.cdf',
        l1c_path / 'imap_ultra_l1c_90sensor-spacecraftpset_20250615-repoint00032_v001.cdf',
        l1c_path / 'imap_ultra_l1c_90sensor-spacecraftpset_20250715-repoint00062_v001.cdf',
        l1c_path / 'imap_ultra_l1c_90sensor-spacecraftpset_20250720-repoint00067_v001.cdf'
    ]

    for file in l1c_filepaths:
        process_date = datetime.strptime(file.name.split('_')[-2].split('-')[0], "%Y%m%d")
        l1c_filepath = file
        all_paths.append(str(l1c_filepath))
        glows_l3e_file_path = get_run_local_data_path(
            f'{base_path}/glows_l3e/imap_glows_l3e_survival-probability-ultra_{process_date:%Y%m%d}_v{version_number:03}.cdf'
        )
        glows_l3e_file_path.parent.mkdir(parents=True, exist_ok=True)
        glows_input_path = get_test_instrument_team_data_path('glows/probSur.Imap.Ul.V0_2009.000.dat')
        create_survival_probabilities_file(glows_file_path=glows_input_path, date_for_file=process_date,
                                           cdf_file_path=get_run_local_data_path(glows_l3e_file_path))

        all_paths.append(str(glows_l3e_file_path))

    l2_out_path = get_run_local_data_path(
        f'{base_path}/imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-6mo_{start_date:%Y%m%d}_v{version_number:03}.cdf'
    )
    l2_out_path.parent.mkdir(parents=True, exist_ok=True)

    parent_paths = [file.name for file in l1c_filepaths]
    create_example_ultra_l2_with_parents(l2_out_path, parent_paths)
    all_paths.append(str(l2_out_path))

    return all_paths


def create_example_ultra_l2_with_parents(out_path, parents: list[str]):
    _create_example_ultra_l2_map_with_power_law(out_path=out_path, parents=parents)


if __name__ == "__main__":
    furnish_local_spice()
    paths = create_l1c_and_glows_with_matching_date_range(datetime(2025, 5, 15, 12), datetime(2025, 7, 20, 12))
    for path in paths:
        imap_data_access.upload(path)
