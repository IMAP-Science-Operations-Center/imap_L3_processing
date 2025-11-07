import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imap_data_access
from imap_data_access import ScienceFilePath, ProcessingInputCollection, RepointInput
from imap_processing.spice.repoint import set_global_repoint_table_paths
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3bc.glows_l3bc_dependencies import GlowsL3BCDependencies
from imap_l3_processing.glows.l3bc.models import CRToProcess, ExternalDependencies
from imap_l3_processing.glows.l3bc.utils import get_date_range_of_cr, get_best_ancillary, \
    get_cr_for_date_time
from imap_l3_processing.utils import read_cdf_parents

logger = logging.getLogger(__name__)


class AvailableServerData:
    l3a_files: list[dict]
    l3b_files: dict[int, dict]
    l3c_files: dict[int, dict]


@dataclass
class GlowsL3BCInitializerData:
    external_dependencies: ExternalDependencies
    l3bc_dependencies: list[GlowsL3BCDependencies]
    l3bs_by_cr: dict[int, str]
    l3cs_by_cr: dict[int, str]
    repoint_file_path: Path


class GlowsL3BCInitializer:
    @staticmethod
    def get_crs_to_process(dependencies: ProcessingInputCollection) -> GlowsL3BCInitializerData:
        [repoint_file] = dependencies.get_file_paths(data_type=RepointInput.data_type)
        repoint_downloaded_path = imap_data_access.download(repoint_file)
        set_global_repoint_table_paths([repoint_downloaded_path])

        l3a_query_results = imap_data_access.query(instrument="glows", data_level="l3a", descriptor="hist",
                                                   version="latest")
        l3a_files_names = [Path(l3a_query_result["file_path"]).name for l3a_query_result in l3a_query_results]
        cr_to_l3a_file_names = GlowsL3BCInitializer.group_l3a_by_cr(l3a_files_names)

        l3b_query_result = imap_data_access.query(instrument="glows", data_level="l3b",
                                                  descriptor="ion-rate-profile", version="latest")
        l3c_query_result = imap_data_access.query(instrument="glows", data_level="l3c", descriptor="sw-profile",
                                                  version="latest")

        l3bs_by_cr = {int(result['cr']): Path(result["file_path"]).name for result in l3b_query_result}
        l3cs_by_cr = {int(result['cr']): Path(result["file_path"]).name for result in l3c_query_result}

        uv_anisotropy_query_result = imap_data_access.query(table="ancillary", instrument="glows",
                                                            descriptor="uv-anisotropy-1CR")
        waw_helio_ion_mp_query_result = imap_data_access.query(table="ancillary", instrument="glows",
                                                               descriptor="WawHelioIonMP")
        bad_days_list_query_result = imap_data_access.query(table="ancillary", instrument="glows",
                                                            descriptor="bad-days-list")
        pipeline_settings_query_result = imap_data_access.query(table="ancillary", instrument="glows",
                                                                descriptor="pipeline-settings-l3bcde")

        logger.info("Downloading external dependencies...")

        external_dependencies = ExternalDependencies.fetch_dependencies()

        logger.info("Finished downloading external dependencies")

        if not all([external_dependencies.f107_index_file_path, external_dependencies.omni2_data_path,
                    external_dependencies.lyman_alpha_path]):
            logger.info(f"Found issues with external dependencies, returning {external_dependencies}")

            return GlowsL3BCInitializerData(
                external_dependencies=external_dependencies,
                l3bc_dependencies=[],
                l3bs_by_cr=l3bs_by_cr,
                l3cs_by_cr=l3cs_by_cr,
                repoint_file_path=repoint_downloaded_path
            )

        all_l3bc_dependencies = []
        for cr_number, l3a_files in cr_to_l3a_file_names.items():
            logger.info(f"considering CR {cr_number}")
            cr_start_date, cr_end_date = get_date_range_of_cr(cr_number)

            ancillaries = {
                "uv-anisotropy-1CR": get_best_ancillary(cr_start_date, cr_end_date, uv_anisotropy_query_result),
                "WawHelioIonMP": get_best_ancillary(cr_start_date, cr_end_date, waw_helio_ion_mp_query_result),
                "bad-days-list": get_best_ancillary(cr_start_date, cr_end_date, bad_days_list_query_result),
                "pipeline-settings-l3bcde": get_best_ancillary(cr_start_date, cr_end_date,
                                                               pipeline_settings_query_result),
            }

            if all(ancillaries.values()):
                cr_candidate = CRToProcess(
                    l3a_files,
                    ancillaries["uv-anisotropy-1CR"],
                    ancillaries["WawHelioIonMP"],
                    ancillaries["bad-days-list"],
                    ancillaries["pipeline-settings-l3bcde"],
                    cr_start_date,
                    cr_end_date,
                    cr_number
                )

                print(f"{cr_candidate=}")

                if version := GlowsL3BCInitializer.should_process_cr_candidate(cr_candidate, l3bs_by_cr,
                                                                               external_dependencies):
                    print(f"{version=}")
                    l3bc_dependencies = GlowsL3BCDependencies.download_from_cr_to_process(cr_candidate, version,
                                                                                          external_dependencies,
                                                                                          repoint_downloaded_path)
                    all_l3bc_dependencies.append(l3bc_dependencies)
                else:
                    logger.info(f"decided not to process {cr_candidate}")
            else:
                logger.info(f"Missing ancillary dependencies: {ancillaries}")

        return GlowsL3BCInitializerData(
            external_dependencies=external_dependencies,
            l3bc_dependencies=all_l3bc_dependencies,
            l3bs_by_cr=l3bs_by_cr,
            l3cs_by_cr=l3cs_by_cr,
            repoint_file_path=repoint_downloaded_path
        )

    @staticmethod
    def should_process_cr_candidate(cr_candidate: CRToProcess, l3bs_by_cr: dict[int, str],
                                    external_dependencies: ExternalDependencies) -> Optional[int]:
        if not cr_candidate.buffer_time_has_elapsed_since_cr():
            logger.warning(f"Not enough time has elapsed for cr {cr_candidate.cr_rotation_number}")
            return None

        if not cr_candidate.has_valid_external_dependencies(external_dependencies):
            logger.warning(f"Invalid external dependencies for {cr_candidate.cr_rotation_number}")
            return None

        match l3bs_by_cr.get(cr_candidate.cr_rotation_number):
            case None:
                return 1
            case l3b_file_name:
                l3b_parents = read_cdf_parents(l3b_file_name)
                if not cr_candidate.pipeline_dependency_file_names().issubset(l3b_parents):
                    return int(ScienceFilePath(l3b_file_name).version[1:]) + 1
        return None

    @staticmethod
    def group_l3a_by_cr(l3a_file_names: list[str]) -> dict[int, set[str]]:
        grouped_l3a_by_cr = defaultdict(set)
        for l3a_file_name in l3a_file_names:
            path = imap_data_access.download(l3a_file_name)
            with CDF(str(path)) as cdf:
                epoch = cdf['epoch'][0]

            cr_number = get_cr_for_date_time(epoch)
            grouped_l3a_by_cr[cr_number].add(l3a_file_name)

        return grouped_l3a_by_cr
