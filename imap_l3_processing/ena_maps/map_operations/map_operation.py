import abc

from imap_processing.ena_maps.ena_maps import AbstractSkyMap


class MapOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, map_obj: AbstractSkyMap) -> AbstractSkyMap:
        pass
