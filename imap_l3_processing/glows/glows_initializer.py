from dataclasses import fields
from pathlib import Path

from imap_data_access import query
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.glows.l3bc.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
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


def _should_process(glows_l3b_dependencies: GlowsInitializerAncillaryDependencies) -> bool:
    for field in fields(glows_l3b_dependencies):
        if getattr(glows_l3b_dependencies, field.name) is None:
            return False
    return True
