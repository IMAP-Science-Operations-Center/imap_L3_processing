# xarray merge could be good for appending new data vars with an assertion that the shape
# of the data does not change, this enforces coordinates from the 2 datasets matching up
from xarray import Dataset

from imap_l3_processing.ena_maps.new_map_types import AbstractPointingSetOperation, DerivedPointingSet


class AddPsetData(AbstractPointingSetOperation):
    def __init__(self, data_to_merge: Dataset, vars_to_merge: list[str]):
        self.data_to_merge = data_to_merge
        self.vars_to_merge = vars_to_merge

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        merged_data = pointing_set.data
        for var in self.vars_to_merge:
            merged_data = merged_data.assign({var: self.data_to_merge[var]})
        return DerivedPointingSet(dataset=merged_data,
                                  spice_reference_frame=pointing_set.spice_reference_frame)
