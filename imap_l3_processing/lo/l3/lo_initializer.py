import dataclasses

import imap_data_access

from imap_l3_processing.maps.map_descriptors import MapDescriptorParts, SurvivalCorrection
from imap_l3_processing.maps.map_initializer import MapInitializer


class LoInitializer(MapInitializer):
    def __init__(self):
        glows_query_result = imap_data_access.query(
            instrument='glows',
            data_level='l3e',
            descriptor="survival-probability-lo",
            version="latest"
        )
        self.glows_files_by_repointing = {int(r["repointing"]): r["file_path"] for r in glows_query_result}

        lo_l2_query_result = imap_data_access.query(
            instrument="lo",
            data_level="l2",
            version="latest"
        )

        lo_l3_query_result = imap_data_access.query(
            instrument="lo",
            data_level="l3",
            version="latest"
        )

        super().__init__("lo", lo_l2_query_result, lo_l3_query_result)

    def _collect_glows_psets_by_repoint(self, descriptor: str) -> dict[int, str]:
        return self.glows_files_by_repointing

    def _get_l2_dependencies(self, descriptor: MapDescriptorParts) -> list[MapDescriptorParts]:
        return [dataclasses.replace(descriptor, survival_correction=SurvivalCorrection.NotSurvivalCorrected)]
