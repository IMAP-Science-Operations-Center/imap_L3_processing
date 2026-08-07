from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_l3_processing.glows.descriptors import GLOWS_L3E_DESCRIPTORS

L3E_DATA_LEVEL = "l3e"
REPOINT_PREFIX = "repoint"
CARRINGTON_ROTATION_PREFIX = "cr"


@dataclass
class ProductToReprocess:
    descriptor: str
    repoints: list[str]
    carrington_rotations: list[str]


@dataclass
class ReprocessInfo:
    products_to_reprocess: list[ProductToReprocess]

    @classmethod
    def parse_from_ancillary(cls, path_to_ancillary: Path) -> ReprocessInfo:
        with open(path_to_ancillary) as file:
            parsed_products = [cls._parse_line(line) for line in file if line.strip()]

        return cls([product for data_level, product in parsed_products if data_level == L3E_DATA_LEVEL])

    @staticmethod
    def _parse_line(line: str) -> tuple[str, ProductToReprocess]:
        product_name, _, remainder = line.strip().partition(" ")
        data_level, _, descriptor = product_name.partition("_")

        time_ranges = remainder.replace(",", " ").split()
        repoints = [time_range for time_range in time_ranges if time_range.startswith(REPOINT_PREFIX)]
        carrington_rotations = [time_range for time_range in time_ranges if
                                time_range.startswith(CARRINGTON_ROTATION_PREFIX)]

        return data_level, ProductToReprocess(descriptor, repoints, carrington_rotations)

    def should_reprocess_l3d(self) -> bool:
        product_descriptors = [product.descriptor for product in self.products_to_reprocess]
        return any(set(product_descriptors).intersection(set(GLOWS_L3E_DESCRIPTORS)))
