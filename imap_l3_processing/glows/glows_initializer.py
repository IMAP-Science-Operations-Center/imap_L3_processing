from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from imap_data_access import query, download
from imap_data_access.processing_input import ProcessingInputCollection
from spacepy.pycdf import CDF

from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3bc.l3bc_toolkit.constants import PHISICAL_CONSTANTS
from imap_l3_processing.glows.l3bc.utils import find_unprocessed_carrington_rotations, archive_dependencies


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

        crs_to_process = find_unprocessed_carrington_rotations(l3a_files, l3b_files, glows_ancillary_dependencies)

        zip_file_paths = []

        for cr_to_process in crs_to_process:
            path = archive_dependencies(cr_to_process, version, glows_ancillary_dependencies)
            zip_file_paths.append(path)

        return zip_file_paths


    # def validata_and_initialize_spike(self, processing_input_collection: ProcessingInputCollection) -> list[Path]:
    #     jd_carrington_first = datetime(2009, 12, 7, 4)
    #     carrington_length = timedelta(days=27.2753)
    #     cr_start_dates = [(i * carrington_length) + jd_carrington_first for i in range(0, 500)]
    #
    #     for cr_start_date in cr_start_dates:
    #         l3b_files = query(instrument="glows", descriptor='ion-rate-profile', version="latest", data_level="l3b", start_date=cr_start_date.strftime("%Y%m%d"))
    #         l3b_file_path = l3b_files[0]["file_path"]
    #         l3b_file = download(l3b_file_path)
    #
    #         with CDF(str(l3b_file)) as l3b_cdf:
    #             carrington_start_date = cr_start_date.strftime("%Y%m%d")
    #             carrington_end_date = (cr_start_date + carrington_length).strftime("%Y%m%d")
    #             l3a_files = query(instrument="glows", version="latest", data_level="l3a", start_date=carrington_start_date, end_date=carrington_end_date)
    #
    #             l3a_file_paths =  {f["file_path"] for f in l3a_files}
    #             parent_files = set(l3b_cdf.attrs["Parents"])
    #
    #             if not l3a_file_paths.issubset(parent_files):
    #                 new_version = int(l3b_files[0]["version"][1:]) + 1
    #
    #
    #
    #
    #
    #
    #     l3_b_files =
    #
    #     # Find current CR for input l3A
    #
    #     latest_l3a_file = None



def _should_process(glows_l3b_dependencies: GlowsInitializerAncillaryDependencies) -> bool:
    for field in fields(glows_l3b_dependencies):
        if getattr(glows_l3b_dependencies, field.name) is None:
            return False
    return True
