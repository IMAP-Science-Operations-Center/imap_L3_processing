from imap_processing.ena_maps.ena_maps import AbstractSkyMap

from imap_l3_processing.ena_maps.map_operations.map_operation import MapOperation


class DivideMapOperation(MapOperation):
    def __init__(self, numerators: list[str], denominator: str):
        self.numerators = numerators
        self.denominator = denominator

    def transform(self, skymap: AbstractSkyMap) -> AbstractSkyMap:
        skymap.data_1d = skymap.data_1d.merge(
            skymap.data_1d[self.numerators] / skymap.data_1d[self.denominator],
            overwrite_vars=self.numerators)
        return skymap
