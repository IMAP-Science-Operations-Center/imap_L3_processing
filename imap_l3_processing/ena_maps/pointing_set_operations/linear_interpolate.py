from typing import Union

import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from xarray.core.types import InterpOptions

from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet, AbstractPointingSetOperation


class LinearInterpolatePointingOperation(AbstractPointingSetOperation):
    def __init__(self, coord: CoordNames, interpolation_coordinates: Union[np.ndarray, list],
                 method: InterpOptions = 'linear'):
        self.method = method
        self.coord = coord.value
        self.interpolation_coordinates = interpolation_coordinates

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        return DerivedPointingSet(dataset=pointing_set.data.interp(coords={self.coord: self.interpolation_coordinates},
                                                                   method=self.method),
                                  spice_reference_frame=pointing_set.spice_reference_frame)
