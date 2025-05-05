import abc

from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet


class AbstractPointingSetOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        pass
