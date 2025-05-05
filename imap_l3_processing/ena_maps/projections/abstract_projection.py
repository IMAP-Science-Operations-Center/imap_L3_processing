import abc

from imap_processing.ena_maps.ena_maps import RectangularSkyMap

from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet


class AbstractProjection(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def project_rectangular(self, psets: list[DerivedPointingSet]) -> RectangularSkyMap:
        pass
