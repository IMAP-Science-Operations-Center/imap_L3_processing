from imap_l3_processing.ena_maps.new_map_types import AbstractPointingSetOperation, DerivedPointingSet


class MultiplyVariables(AbstractPointingSetOperation):
    def __init__(self, variables_to_multiply: list[str], multiplier_var: str):
        self.variables_to_multiply = variables_to_multiply
        self.multiplier_var = multiplier_var

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        weighted_dataset = pointing_set.data[self.variables_to_multiply] * pointing_set.data[self.multiplier_var]
        new_data = pointing_set.data.merge(weighted_dataset, overwrite_vars=self.variables_to_multiply)
        return DerivedPointingSet(dataset=new_data, spice_reference_frame=pointing_set.spice_reference_frame)
