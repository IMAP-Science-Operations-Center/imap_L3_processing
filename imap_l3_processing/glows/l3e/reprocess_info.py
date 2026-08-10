from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imap_data_access

from imap_l3_processing.glows.descriptors import GLOWS_L3E_DESCRIPTORS, GLOWS_REPROCESSING_DESCRIPTOR
from imap_l3_processing.glows.l3e.glows_l3e_utils import get_repoint_numbers_within_cr_window

L3E_DATA_LEVEL = "l3e"
REPOINT_PREFIX = "repoint"
CARRINGTON_ROTATION_PREFIX = "cr"


@dataclass
class ReprocessTargets:
    repoints: list[int]
    carrington_rotations: list[int]


@dataclass
class ReprocessInfo:
    products_to_reprocess: dict[str, ReprocessTargets]

    @classmethod
    def parse_from_ancillary(cls, path_to_ancillary: Path) -> ReprocessInfo:
        with open(path_to_ancillary) as file:
            parsed_products = [cls._parse_line(line) for line in file if line.strip()]

        return cls({descriptor: target for data_level, descriptor, target in parsed_products if data_level == L3E_DATA_LEVEL})

    @staticmethod
    def _parse_line(line: str) -> tuple[str, str, ReprocessTargets]:
        product_name, _, remainder = line.strip().partition(" ")
        data_level, _, descriptor = product_name.partition("_")

        targets = remainder.replace(",", " ").split()
        repoints = [int(target.removeprefix(REPOINT_PREFIX)) for target in targets if target.startswith(REPOINT_PREFIX)]
        carrington_rotations = [int(target.removeprefix(CARRINGTON_ROTATION_PREFIX)) for target in targets if
                                target.startswith(CARRINGTON_ROTATION_PREFIX)]

        return data_level, descriptor, ReprocessTargets(repoints, carrington_rotations)

    def should_reprocess_l3d(self) -> bool:
        return any(
            set(self.products_to_reprocess.keys()).intersection(
                set(GLOWS_L3E_DESCRIPTORS)
            )
        )

    def get_repoints_for_descriptor(
        self, descriptor: str, repointing_data
    ) -> list[int]:
        reprocess_targets = self.products_to_reprocess.get(descriptor, None)
        if reprocess_targets is None:
            return []

        repoints = reprocess_targets.repoints

        for cr in reprocess_targets.carrington_rotations:
            repoints += get_repoint_numbers_within_cr_window(cr, cr, repointing_data)

        return repoints

def fetch_reprocess_info():
    ancillary_file = imap_data_access.query(instrument='glows', version="latest", descriptor=GLOWS_REPROCESSING_DESCRIPTOR, table="ancillary")[0]
    path_to_downloaded_ancillary = imap_data_access.download(Path(ancillary_file["file_path"]).name)
    return ReprocessInfo.parse_from_ancillary(path_to_downloaded_ancillary)