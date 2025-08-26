import logging
from collections import defaultdict
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ScienceFilePath

from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import \
    GlowsInitializerAncillaryDependencies, F107_FLUX_TABLE_URL, LYMAN_ALPHA_COMPOSITE_INDEX_URL, OMNI2_URL
from imap_l3_processing.glows.l3bc.models import CRToProcess
from imap_l3_processing.glows.l3bc.utils import get_pointing_date_range, get_date_range_of_cr, get_best_ancillary, \
    read_cdf_parents, \
    get_cr_for_date_time
from imap_l3_processing.utils import download_external_dependency

logger = logging.getLogger(__name__)


class GlowsInitializer:

    @staticmethod
    def get_crs_to_process(l3bs_by_cr: dict[int, str]):
        l3a_query_results = imap_data_access.query(instrument="glows", data_level="l3a", version="latest")
        l3a_files_names = [Path(l3a_query_result["file_path"]).name for l3a_query_result in l3a_query_results]
        cr_to_l3a_file_names = GlowsInitializer.group_l3a_by_cr(l3a_files_names)

        uv_anisotropy_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR")
        waw_helio_ion_mp_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="WawHelioIonMP")
        bad_days_list_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="bad-days-list")
        pipeline_settings_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde")

        f107_index_file_path = download_external_dependency(F107_FLUX_TABLE_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
        lyman_alpha_path = download_external_dependency(LYMAN_ALPHA_COMPOSITE_INDEX_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')
        omni2_data_path = download_external_dependency(OMNI2_URL, TEMP_CDF_FOLDER_PATH / 'f107_fluxtable.txt')

        if not all([f107_index_file_path, lyman_alpha_path, omni2_data_path]):
            return []

        _comment_headers(f107_index_file_path)

        crs_to_process = []
        for cr_number, l3a_files in cr_to_l3a_file_names.items():
            cr_start_date, cr_end_date = get_date_range_of_cr(cr_number)

            uv_anisotropy_file_name = get_best_ancillary(cr_start_date, cr_end_date, uv_anisotropy_query_result)
            waw_helio_ion_mp_file_name = get_best_ancillary(cr_start_date, cr_end_date, waw_helio_ion_mp_query_result)
            bad_days_list_file_name = get_best_ancillary(cr_start_date, cr_end_date, bad_days_list_query_result)
            pipeline_settings_file_name = get_best_ancillary(cr_start_date, cr_end_date, pipeline_settings_query_result)

            if all([uv_anisotropy_file_name, waw_helio_ion_mp_file_name, bad_days_list_file_name, pipeline_settings_file_name]):
                cr_candidate = CRToProcess(
                    l3a_files,
                    uv_anisotropy_file_name,
                    waw_helio_ion_mp_file_name,
                    bad_days_list_file_name,
                    pipeline_settings_file_name,
                    f107_index_file_path,
                    lyman_alpha_path,
                    omni2_data_path,
                    cr_start_date,
                    cr_end_date,
                    cr_number
                )

                if version := GlowsInitializer.should_process_cr_candidate(cr_candidate, l3bs_by_cr):
                    crs_to_process.append((version, cr_candidate))

        return crs_to_process

    @staticmethod
    def should_process_cr_candidate(cr_candidate: CRToProcess, l3bs_by_cr: dict[int, str]) -> Optional[int]:
        if not cr_candidate.buffer_time_has_elapsed_since_cr():
            return None

        if not cr_candidate.has_valid_external_dependencies():
            return None

        match l3bs_by_cr.get(cr_candidate.cr_rotation_number):
            case None:
                return 1
            case l3b_file_name:
                l3b_parents = read_cdf_parents(l3b_file_name)
                if not cr_candidate.pipeline_dependency_file_names().issubset(l3b_parents):
                    return int(ScienceFilePath(l3b_file_name).version[1:]) + 1

    @staticmethod
    def group_l3a_by_cr(l3a_file_paths: list[str]) -> dict[int, set[str]]:
        grouped_l3a_by_cr = defaultdict(set)
        for l3a_file_path in l3a_file_paths:
            start, end = get_pointing_date_range(ScienceFilePath(l3a_file_path).repointing)
            start_cr = get_cr_for_date_time(start.astype(datetime))
            end_cr = get_cr_for_date_time(end.astype(datetime))

            grouped_l3a_by_cr[start_cr].add(l3a_file_path)
            grouped_l3a_by_cr[end_cr].add(l3a_file_path)

        return grouped_l3a_by_cr


def _comment_headers(filename: Path, num_lines=2):
    with open(filename, "r+") as file:
        lines = file.readlines()
        for i in range(num_lines):
            lines[i] = "#" + lines[i]
        file.truncate(0)
    with open(filename, "w") as file:
        file.writelines(lines)
