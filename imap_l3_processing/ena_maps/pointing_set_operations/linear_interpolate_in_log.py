from typing import Union

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames

from imap_l3_processing.ena_maps.new_map_types import AbstractPointingSetOperation, DerivedPointingSet


class LinearInterpolateInLogOperation(AbstractPointingSetOperation):
    def __init__(self, coord: CoordNames, interpolation_coordinates: Union[np.ndarray, list]):
        self.coord = coord.value
        self.interpolation_coordinates = interpolation_coordinates

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        assert self.coord in pointing_set.data

        pset_dims = pointing_set.data[self.coord].dims
        log_pset_values = np.log10(pointing_set.data[self.coord].values)

        pointing_set.data = pointing_set.data.assign_coords(
            {self.coord: (pset_dims, log_pset_values)})

        pointing_set.data = pointing_set.data.interp(coords={self.coord: np.log10(self.interpolation_coordinates)})
        pointing_set.data = pointing_set.data.assign_coords(
            {self.coord: (pset_dims, 10 ** pointing_set.data[self.coord].values)})

        return pointing_set
