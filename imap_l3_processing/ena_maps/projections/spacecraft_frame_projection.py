import numpy as np
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.ena_maps.new_map_types import DerivedPointingSet, RectangularProtomap
from imap_l3_processing.ena_maps.projections.abstract_projection import AbstractProjection


class SpacecraftFrameProjection(AbstractProjection):
    def __init__(self, spice_frame: SpiceFrame, vars_to_project: list[str]):
        self.vars_to_project = vars_to_project
        self.spice_frame = spice_frame

    def project_rectangular(self, psets: list[DerivedPointingSet], spacing_degree: int = 4) -> RectangularProtomap:
        skymap = RectangularProtomap(spacing_deg=spacing_degree, spice_frame=self.spice_frame)
        for pset in psets:
            assert CoordNames.AZIMUTH_L1C.value in pset.data
            assert CoordNames.ELEVATION_L1C.value in pset.data

            azimuths = pset.data[CoordNames.AZIMUTH_L1C.value].values
            elevations = pset.data[CoordNames.ELEVATION_L1C.value].values

            first_var_to_project = self.vars_to_project[0]

            pset.spatial_coords = [CoordNames.AZIMUTH_L1C.value, CoordNames.AZIMUTH_L1C.value,
                                   CoordNames.ELEVATION_L1C.value]
            num_points = np.prod([len(pset.data[dim].values) for dim in pset.data[first_var_to_project].dims if
                                  dim in pset.spatial_coords])

            pset.az_el_points = np.column_stack([azimuths, elevations])
            pset.num_points = num_points
            skymap.project_pset_values_to_map(pset, self.vars_to_project)
        return skymap
