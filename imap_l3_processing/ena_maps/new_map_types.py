from typing import Self

import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet, AbstractSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.ena_maps.cdf_map_format import MapVars, IntensityMapVars
from imap_l3_processing.ena_maps.map_operations.divide_map_operation import DivideMapOperation
from imap_l3_processing.ena_maps.map_operations.map_operation import MapOperation
from imap_l3_processing.ena_maps.pointing_set_operations.abstract_pointing_set_operation import \
    AbstractPointingSetOperation
from imap_l3_processing.ena_maps.pointing_set_operations.linear_interpolate import LinearInterpolatePointingOperation
from imap_l3_processing.ena_maps.pointing_set_operations.linear_interpolate_in_log import \
    LinearInterpolateInLogOperation
from imap_l3_processing.ena_maps.pointing_set_operations.mask_var_by_spin_phase import MaskVarBySpinPhase
from imap_l3_processing.ena_maps.pointing_set_operations.weight_pset import MultiplyVariables
from imap_l3_processing.ena_maps.projections.spacecraft_frame_projection import SpacecraftFrameProjection
from imap_l3_processing.hi.l3.utils import SpinPhase, Sensor


class DerivedPointingSet(PointingSet):
    def __init__(self, dataset: xr.Dataset, spice_reference_frame: geometry.SpiceFrame):
        super().__init__(dataset, spice_reference_frame)
        self.data = dataset

    def __rshift__(self, transform: AbstractPointingSetOperation) -> Self:
        return transform.transform(self)

    def apply(self, operation: AbstractPointingSetOperation) -> Self:
        return operation.transform(self)


class RectangularProtomap(RectangularSkyMap):
    def __init__(self, spacing_deg: int, spice_frame=SpiceFrame) -> None:
        super().__init__(spacing_deg, spice_frame)

    def apply(self, operation: MapOperation) -> Self:
        return operation.transform(self)


def calculate_l1c_spin_angles(num_bins: int) -> np.ndarray:
    deg_spacing = 360 / num_bins
    half_bin_width = deg_spacing / 2
    spin_angles = np.linspace(0, 360, num_bins,
                              endpoint=False) + half_bin_width
    return np.mod(spin_angles + 90, 360)


class HiL1CBinsToPhysicalUnits(AbstractPointingSetOperation):
    def __init__(self, l1c_sensor: Sensor, l2_energies: np.ndarray):
        self.l2_energies = l2_energies
        self.l1c_sensor = l1c_sensor

    def transform(self, pointing_set: DerivedPointingSet) -> DerivedPointingSet:
        num_l1c_spatial_bins = len(pointing_set.data["spin_angle_bin"].values)

        pointing_set.data = pointing_set.data.rename(
            {'spin_angle_bin': CoordNames.AZIMUTH_L1C.value, 'esa_energy_step': CoordNames.ENERGY.value})

        elevations = np.repeat(Sensor.get_sensor_angle(self.l1c_sensor.value), num_l1c_spatial_bins)

        pointing_set.data[CoordNames.ELEVATION_L1C.value] = elevations
        pointing_set.data[CoordNames.AZIMUTH_L1C.value] = calculate_l1c_spin_angles(num_l1c_spatial_bins)
        pointing_set.data[CoordNames.ENERGY.value] = self.l2_energies

        return pointing_set


"""
# option 0
physical_l1c = physical_transform.transform(l1c_pset)
masked_l1c = spin_phase_mask.transform(physical_l1c)

# option 1
l1c_pset >> physical_transform >> spin_phase_mask

# option 3
composed_l1c = pipe(l1c_pointing_set_transforms)

l1c_pset.pipe()

# ----
transform_glows_psets(glows_pset)
"""


class HiSurvivalCorrection:
    def survival_correct(self, hi_sensor: Sensor,
                         l1c_psets: list[DerivedPointingSet],
                         l2_map: RectangularProtomap,
                         glows_psets: list[DerivedPointingSet]) -> AbstractSkyMap:
        # l1c and glows would be lined up prior

        interpolate_to_map_energies = LinearInterpolateInLogOperation(CoordNames.ENERGY,
                                                                      l2_map.data_1d[CoordNames.ENERGY.value].values),

        physical_transform = HiL1CBinsToPhysicalUnits(hi_sensor, l2_map.data_1d[CoordNames.ENERGY.value].values)
        mask_exposure_by_spin_phase = MaskVarBySpinPhase(MapVars.EXPOSURE_FACTOR, SpinPhase.RamOnly),

        sp_var_name = "probability_of_survival"

        derived_psets = []
        for l1c_pset, glows_pset in zip(l1c_psets, glows_psets):
            derived_l1c_pset = l1c_pset \
                .apply(physical_transform) \
                .apply(mask_exposure_by_spin_phase)

            interpolate_to_l1c_spin_angles = LinearInterpolatePointingOperation(CoordNames.AZIMUTH_L1C,
                                                                                derived_l1c_pset.data[
                                                                                    CoordNames.AZIMUTH_L1C.value],
                                                                                method="nearest")
            derived_glows_pset = glows_pset \
                .apply(interpolate_to_map_energies) \
                .apply(interpolate_to_l1c_spin_angles)

            combined_pset = combine_pset_data(derived_l1c_pset.data[[sp_var_name]],
                                              derived_glows_pset.data[
                                                  [MapVars.EXPOSURE_FACTOR, CoordNames.ELEVATION_L1C.value]])

            exposure_weight_survivals = MultiplyVariables(variables_to_multiply=[sp_var_name],
                                                          multiplier_var=MapVars.EXPOSURE_FACTOR)
            weighted_survival_pset = combined_pset.apply(exposure_weight_survivals)
            derived_psets.append(weighted_survival_pset)

        sc_projection = SpacecraftFrameProjection(SpiceFrame.ECLIPJ2000, [sp_var_name, MapVars.EXPOSURE_FACTOR])
        sp_skymap = sc_projection.project_rectangular(derived_psets, 4)

        divide_out_weights = DivideMapOperation([sp_var_name], MapVars.EXPOSURE_FACTOR)
        sp_skymap = sp_skymap.apply(divide_out_weights)

        # would have to combine maps here

        survival_correct = DivideMapOperation([
            IntensityMapVars.ENA_INTENSITY,
            IntensityMapVars.ENA_INTENSITY_STAT_UNC,
            IntensityMapVars.ENA_INTENSITY_SYS_ERR
        ], sp_var_name)
        sp_corrected_l2_map = l2_map.apply(survival_correct)

        return sp_corrected_l2_map
