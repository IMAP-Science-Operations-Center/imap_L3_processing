from dataclasses import fields

from imap_data_access import query

from imap_l3_processing.glows.l3b.glows_initializer_ancillary_dependencies import GlowsInitializerAncillaryDependencies
from imap_l3_processing.glows.l3b.utils import find_unprocessed_carrington_rotations


class GlowsInitializer:
    @staticmethod
    def validate_and_initialize(version: str):
        glows_ancillary_dependencies = GlowsInitializerAncillaryDependencies.fetch_dependencies()
        if not _should_process(glows_ancillary_dependencies):
            return []
        l3a_files = query(instrument="glows", version=version, data_level="l3a")
        l3b_files = query(instrument="glows", version=version, data_level="l3b")

        find_unprocessed_carrington_rotations(l3a_files, l3b_files, glows_ancillary_dependencies.omni2_data_path,
                                              glows_ancillary_dependencies.f107_index_file_path,
                                              glows_ancillary_dependencies.lyman_alpha_path)


def _should_process(glows_l3b_dependencies: GlowsInitializerAncillaryDependencies) -> bool:
    for field in fields(glows_l3b_dependencies):
        if getattr(glows_l3b_dependencies, field.name) is None:
            return False
    return True
