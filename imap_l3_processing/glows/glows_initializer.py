import json
import logging
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import imap_data_access
from astropy.time import TimeDelta, Time
from imap_data_access import query, ScienceFilePath
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput
from imap_processing.spice.repoint import set_global_repoint_table_paths
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3bc.dependency_validator import validate_dependencies
from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.l3bc_toolkit.funcs import carrington, jd_fm_Carrington
from imap_l3_processing.glows.l3bc.models import CRToProcess, L3BCAncillaryQueryResults, read_pipeline_settings
from imap_l3_processing.glows.l3bc.utils import archive_dependencies, \
    get_astropy_time_from_yyyymmdd, get_pointing_date_range, get_date_range_of_cr, get_best_ancillary, read_cdf_parents, \
    get_cr_for_date_time

logger = logging.getLogger(__name__)


class GlowsInitializer:
    @staticmethod
    def validate_and_initialize(version: str, processing_input_collection: ProcessingInputCollection) -> list[Path]:
        glows_ancillary_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies(
            processing_input_collection)
        if not _should_process(glows_ancillary_dependencies):
            return []
        input_l3a_version = processing_input_collection.get_science_inputs('glows')[0].imap_file_paths[0].version

        l3a_files = query(instrument="glows", descriptor="hist", version=input_l3a_version, data_level="l3a")
        l3b_files = query(instrument="glows", descriptor='ion-rate-profile', version=version, data_level="l3b")
        logger.info(f"l3a files {[f["file_path"] for f in l3a_files]}")
        logger.info(f"l3b files {[f["file_path"] for f in l3b_files]}")

        crs_to_process = GlowsInitializer.find_unprocessed_carrington_rotations(l3a_files, l3b_files, glows_ancillary_dependencies)

        zip_file_paths = []
        logger.info(f"making zips for crs: {[cr.cr_rotation_number for cr in crs_to_process]}")

        for cr_to_process in crs_to_process:
            path = archive_dependencies(cr_to_process, version, glows_ancillary_dependencies)
            zip_file_paths.append(path)

        return zip_file_paths

    @staticmethod
    def find_unprocessed_carrington_rotations(l3a_inputs: list[dict], l3b_inputs: list[dict],
                                              dependencies: GlowsInitializerAncillaryDependencies) -> [CRToProcess]:
        l3bs_carringtons: set = set()
        for l3b in l3b_inputs:
            current_date = get_astropy_time_from_yyyymmdd(l3b["start_date"]) + TimeDelta(1, format='jd')
            current_rounded_cr = int(carrington(current_date.jd))
            l3bs_carringtons.add(current_rounded_cr)

        sorted_l3a_inputs = sorted(l3a_inputs, key=lambda entry: entry['start_date'])

        l3as_by_carrington: dict = defaultdict(list)

        latest_l3a_file = get_astropy_time_from_yyyymmdd(sorted_l3a_inputs[-1]["start_date"])
        set_global_repoint_table_paths([dependencies.repointing_file])
        for index, l3a in enumerate(sorted_l3a_inputs):
            repointing_start, repointing_end = get_pointing_date_range(l3a["repointing"])
            start_cr = int(carrington(Time(repointing_start, format="datetime64").jd))
            end_cr = int(carrington(Time(repointing_end, format="datetime64").jd))

            if end_cr - start_cr == 1:
                l3as_by_carrington[end_cr].append(l3a['file_path'])

            l3as_by_carrington[start_cr].append(l3a['file_path'])

        crs_to_process = []
        for carrington_number, l3a_files in l3as_by_carrington.items():
            if carrington_number not in l3bs_carringtons:
                carrington_start_date = jd_fm_Carrington(float(carrington_number))
                date_time = Time(carrington_start_date, format='jd')
                date_time.format = 'iso'
                carrington_end_date_non_inclusive = jd_fm_Carrington(carrington_number + 1)
                date_time_end_date = Time(carrington_end_date_non_inclusive, format='jd')
                date_time_end_date.format = 'iso'

                if latest_l3a_file < date_time_end_date + dependencies.initializer_time_buffer:
                    continue

                is_valid = validate_dependencies(date_time_end_date, dependencies.initializer_time_buffer,
                                                 dependencies.omni2_data_path, dependencies.f107_index_file_path,
                                                 dependencies.lyman_alpha_path)

                if not is_valid:
                    continue

                crs_to_process.append(CRToProcess(
                    l3a_paths=l3a_files,
                    cr_start_date=date_time,
                    cr_end_date=date_time_end_date,
                    cr_rotation_number=carrington_number,
                ))

        return crs_to_process

    @staticmethod
    def get_crs_to_process():
        l3a_query_results = imap_data_access.query(instrument="glows", data_level="l3a", version="latest")
        l3a_files_names = [Path(l3a_query_result["file_path"]).name for l3a_query_result in l3a_query_results]
        cr_to_l3a_file_names = GlowsInitializer.group_l3a_by_cr(l3a_files_names)

        uv_anisotropy_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="uv-anisotropy-1CR")
        waw_helio_ion_mp_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="WawHelioIonMP")
        bad_days_list_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="bad-days-list")
        pipeline_settings_query_result = imap_data_access.query(table="ancillary", instrument="glows", descriptor="pipeline-settings-l3bcde")

        cr_candidates = []
        for cr_number, l3a_files in cr_to_l3a_file_names.items():
            cr_start_date, cr_end_date = get_date_range_of_cr(cr_number)

            uv_anisotropy_file_name = get_best_ancillary(cr_start_date, cr_end_date, uv_anisotropy_query_result)
            waw_helio_ion_mp_file_name = get_best_ancillary(cr_start_date, cr_end_date, waw_helio_ion_mp_query_result)
            bad_days_list_file_name = get_best_ancillary(cr_start_date, cr_end_date, bad_days_list_query_result)
            pipeline_settings_file_name = get_best_ancillary(cr_start_date, cr_end_date, pipeline_settings_query_result)

            if all([uv_anisotropy_file_name, waw_helio_ion_mp_file_name, bad_days_list_file_name, pipeline_settings_file_name]):
                cr_candidates.append(CRToProcess(
                    l3a_files,
                    uv_anisotropy_file_name,
                    waw_helio_ion_mp_file_name,
                    bad_days_list_file_name,
                    pipeline_settings_file_name,
                    cr_start_date,
                    cr_end_date,
                    cr_number
                ))

        crs_to_process = []
        for cr_candidate in cr_candidates:
            if version := GlowsInitializer.should_process_cr_candidate(cr_candidate):
                crs_to_process.append((version, cr_candidate))

        return crs_to_process

    @staticmethod
    def should_process_cr_candidate(cr_candidate: CRToProcess) -> Optional[int]:
        if not cr_candidate.buffer_time_has_elapsed_since_cr():
            return None

        start_date_for_query = cr_candidate.cr_start_date.strftime("%Y%m%d")
        end_date_for_query = cr_candidate.cr_end_date.strftime("%Y%m%d")

        l3b_query_result = imap_data_access.query(
            instrument="glows",
            data_level="l3b",
            descriptor="ion-rate-profile",
            start_date=start_date_for_query,
            end_date=end_date_for_query,
            version="latest"
        )

        match l3b_query_result:
            case []:
                return 1
            case [existing_l3b]:
                l3b_file_name = Path(existing_l3b["file_path"]).name
                l3b_parents = read_cdf_parents(l3b_file_name)
                if not cr_candidate.pipeline_dependency_file_names().issubset(l3b_parents):
                    return int(ScienceFilePath(l3b_file_name).version[1:]) + 1
            case _:
                raise ValueError(f"Expected to find up to one latest L3b file on the server for the date range: {start_date_for_query} to {end_date_for_query}!")

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

def _should_process(glows_l3b_dependencies: GlowsInitializerAncillaryDependencies) -> bool:
    for field in fields(glows_l3b_dependencies):
        if getattr(glows_l3b_dependencies, field.name) is None:
            return False
    return True
