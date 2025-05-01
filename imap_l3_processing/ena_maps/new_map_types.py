import abc
from typing import Union, Self

import numpy as np
import xarray
import xarray as xr
from cdflib.xarray import cdf_to_xarray
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet, AbstractSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame
from xarray.core.types import InterpOptions

from imap_l3_processing.hi.l3.utils import SpinPhase, Sensor


class AbstractPointingSet(PointingSet):
    def __init__(self, dataset: xr.Dataset, spice_reference_frame: geometry.SpiceFrame):
        super().__init__(dataset, spice_reference_frame)
        self.data = dataset

    @classmethod
    def from_cdf(cls, file_path: str, spice_frame: SpiceFrame) -> Self:
        return cls(cdf_to_xarray(file_path, to_datetime=False), spice_frame)

    def __rshift__(self, transform: 'PointingSetOperation') -> Self:
        return transform.transform(self)


class GlowsPointingSet(AbstractPointingSet):
    @classmethod
    def from_cdf(cls, file_path: str, spice_frame: SpiceFrame) -> Self:
        return cls(
            cdf_to_xarray(file_path, to_datetime=False).rename({"spin_angle": CoordNames.AZIMUTH_L1C.value}),
            spice_frame)


class PointingSetOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        pass


class AbstractProjection(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def project_rectangular(self, psets: list[AbstractPointingSet]) -> RectangularSkyMap:
        pass


class SpacecraftFrameProjection(AbstractProjection):
    def __init__(self, spice_frame: SpiceFrame, vars_to_project: list[str]):
        self.vars_to_project = vars_to_project
        self.spice_frame = spice_frame

    def project_rectangular(self, psets: list[AbstractPointingSet], spacing_degree: int = 4) -> RectangularSkyMap:
        skymap = RectangularSkyMap(spacing_deg=spacing_degree, spice_frame=self.spice_frame)
        for pset in psets:
            assert CoordNames.AZIMUTH_L1C.value in pset.data
            assert CoordNames.ELEVATION_L1C.value in pset.data

            azimuths = pset.data[CoordNames.AZIMUTH_L1C.value].values
            elevations = pset.data[CoordNames.ELEVATION_L1C.value].values

            first_var_to_project = self.vars_to_project[0]

            pset.spatial_coords = [CoordNames.AZIMUTH_L1C.value, CoordNames.ELEVATION_L1C.value]
            num_points = np.prod([len(pset.data[dim].values) for dim in pset.data[first_var_to_project].dims if
                                  dim in pset.spatial_coords])

            pset.az_el_points = np.column_stack([azimuths, elevations])
            pset.num_points = num_points
            skymap.project_pset_values_to_map(pset, self.vars_to_project)
        return skymap


class MapOperation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, map_obj: AbstractSkyMap) -> AbstractSkyMap:
        pass


class DivideOutWeights(MapOperation):
    def __init__(self, variables_to_weight: list[str], weight_var: str):
        self.variables_to_weight = variables_to_weight
        self.exposure_variable = weight_var

    def transform(self, skymap: AbstractSkyMap) -> AbstractSkyMap:
        skymap.data_1d = skymap.data_1d.merge(
            skymap.data_1d[self.variables_to_weight] / skymap.data_1d[self.exposure_variable],
            overwrite_vars=self.variables_to_weight)
        return skymap


class SurvivalProbabilityCorrection(MapOperation):
    def __init__(self, survival_probabilities: np.ndarray):
        self.survival_probabilities = survival_probabilities

    def transform(self, skymap: AbstractSkyMap) -> AbstractSkyMap:
        variables_to_correct = [
            "ena_intensity",
            "ena_intensity_stat_unc",
            "ena_intensity_sys_err"
        ]
        skymap.data_1d = skymap.data_1d.merge(skymap.data_1d[variables_to_correct] / self.survival_probabilities,
                                              overwrite_vars=variables_to_correct)
        return skymap


class LinearInterpolateInLogOperation(PointingSetOperation):
    def __init__(self, coord: CoordNames, interpolation_coordinates: Union[np.ndarray, list]):
        self.coord = coord.value
        self.interpolation_coordinates = interpolation_coordinates

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        assert self.coord in pointing_set.data

        pset_dims = pointing_set.data[self.coord].dims
        log_pset_values = np.log10(pointing_set.data[self.coord].values)

        pointing_set.data = pointing_set.data.assign_coords(
            {self.coord: (pset_dims, log_pset_values)})

        pointing_set.data = pointing_set.data.interp(coords={self.coord: np.log10(self.interpolation_coordinates)})
        pointing_set.data = pointing_set.data.assign_coords(
            {self.coord: (pset_dims, 10 ** pointing_set.data[self.coord].values)})

        return pointing_set


class LinearInterpolatePointingOperation(PointingSetOperation):
    def __init__(self, coord: CoordNames, interpolation_coordinates: Union[np.ndarray, list],
                 method: InterpOptions = 'linear'):
        self.method = method
        self.coord = coord.value
        self.interpolation_coordinates = interpolation_coordinates

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        return AbstractPointingSet(dataset=pointing_set.data.interp(coords={self.coord: self.interpolation_coordinates},
                                                                    method=self.method),
                                   spice_reference_frame=pointing_set.spice_reference_frame)


class MaskVarBySpinPhase(PointingSetOperation):
    def __init__(self, var_to_mask: str, spin_phase: SpinPhase):
        assert spin_phase in [SpinPhase.RamOnly, SpinPhase.RamOnly]

        self.var_to_mask = var_to_mask
        self.spin_phase = spin_phase

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        pset_azimuths = calculate_l1c_spin_angles(3600)
        antiram_mask = np.less(90, pset_azimuths) & np.less(pset_azimuths, 270)
        ram_mask = ~antiram_mask

        phase_to_mask = {SpinPhase.RamOnly: ram_mask, SpinPhase.AntiRamOnly: antiram_mask}
        masked_data = pointing_set.data[self.var_to_mask].values * phase_to_mask[self.spin_phase]
        return AbstractPointingSet(dataset=pointing_set.data.assign(
            {self.var_to_mask: (pointing_set.data[self.var_to_mask].dims, masked_data)}),
            spice_reference_frame=pointing_set.spice_reference_frame)


# xarray merge could be good for appending new data vars with an assertion that the shape
# of the data does not change, this enforces coordinates from the 2 datasets matching up
class AddPsetData(PointingSetOperation):
    def __init__(self, data_to_merge: xarray.Dataset, vars_to_merge: list[str]):
        self.data_to_merge = data_to_merge
        self.vars_to_merge = vars_to_merge

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        merged_data = pointing_set.data
        for var in self.vars_to_merge:
            merged_data = merged_data.assign({var: self.data_to_merge[var]})
        return AbstractPointingSet(dataset=merged_data,
                                   spice_reference_frame=pointing_set.spice_reference_frame)


class WeightPointingSet(PointingSetOperation):
    def __init__(self, variables_to_weight: list[str], weight_var: str):
        self.variables_to_weight = variables_to_weight
        self.weight_var = weight_var

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        weighted_dataset = pointing_set.data[self.variables_to_weight] * pointing_set.data[self.weight_var]
        new_data = pointing_set.data.merge(weighted_dataset, overwrite_vars=self.variables_to_weight)
        return AbstractPointingSet(dataset=new_data, spice_reference_frame=pointing_set.spice_reference_frame)


def calculate_l1c_spin_angles(num_bins: int) -> np.ndarray:
    deg_spacing = 360 / num_bins
    half_bin_width = deg_spacing / 2
    spin_angles = np.linspace(0, 360, num_bins,
                              endpoint=False) + half_bin_width
    return np.mod(spin_angles + 90, 360)


class HiL1CBinsToPhysicalUnits(PointingSetOperation):
    def __init__(self, l1c_sensor: Sensor, l2_energies: np.ndarray):
        self.l2_energies = l2_energies
        self.l1c_sensor = l1c_sensor

    def transform(self, pointing_set: AbstractPointingSet) -> AbstractPointingSet:
        num_l1c_spatial_bins = len(pointing_set.data["spin_angle_bin"].values)

        pointing_set.data = pointing_set.data.rename(
            {'spin_angle_bin': CoordNames.AZIMUTH_L1C.value, 'esa_energy_step': CoordNames.ENERGY.value})

        elevations = np.repeat(Sensor.get_sensor_angle(self.l1c_sensor.value), num_l1c_spatial_bins)

        pointing_set.data[CoordNames.ELEVATION_L1C.value] = elevations
        pointing_set.data[CoordNames.AZIMUTH_L1C.value] = calculate_l1c_spin_angles(num_l1c_spatial_bins)
        pointing_set.data[CoordNames.ENERGY.value] = self.l2_energies

        return pointing_set


def pipe(ops: list[PointingSetOperation]):
    def x(pset: PointingSet):
        result = pset
        for op in ops:
            result = op.transform(result)
        return result

    return x


class HiSPCorrectedMap:
    def create_map(self, hi_sensor: Sensor, l1c_psets: list[AbstractPointingSet], l2: RectangularSkyMap,
                   glows_psets: list[GlowsPointingSet]) -> AbstractSkyMap:
        # l1c and glows would be lined up

        glows_pointing_set_transforms = [
            LinearInterpolateInLogOperation(CoordNames.ENERGY, l2.data_1d[CoordNames.ENERGY.value].values),
            LinearInterpolatePointingOperation(CoordNames.AZIMUTH_L1C, calculate_l1c_spin_angles(3600),
                                               method="nearest")
        ]

        transform_glows_psets = pipe([
            LinearInterpolateInLogOperation(CoordNames.ENERGY, l2.data_1d[CoordNames.ENERGY.value].values),
            LinearInterpolatePointingOperation(CoordNames.AZIMUTH_L1C, calculate_l1c_spin_angles(3600),
                                               method="nearest")
        ])

        exposure_factor_var_name = "exposure_factor"
        l1c_pointing_set_transforms = [
            HiL1CBinsToPhysicalUnits(hi_sensor, l2.data_1d[CoordNames.ENERGY.value].values),
            MaskVarBySpinPhase(exposure_factor_var_name, SpinPhase.RamOnly),
        ]

        sp_var_name = "probability_of_survival"
        merged_pointing_set_transforms = [
            WeightPointingSet(variables_to_weight=[sp_var_name], weight_var=exposure_factor_var_name)
        ]

        derived_psets = []
        for l1c_pset, glows_pset in zip(l1c_psets, glows_psets):
            physical_transform, spin_phase_mask = l1c_pointing_set_transforms
            # option 0
            physical_l1c = physical_transform.transform(l1c_pset)
            masked_l1c = spin_phase_mask.transform(physical_l1c)

            # option 1
            for t in [
                HiL1CBinsToPhysicalUnits(hi_sensor, l2.data_1d[CoordNames.ENERGY.value].values),
                MaskVarBySpinPhase(exposure_factor_var_name, SpinPhase.RamOnly),
            ]:
                l1c_pset = l1c_pset >> t
            # option 2
            l1c_pset >> physical_transform >> spin_phase_mask

            # option 3
            composed_l1c = pipe(l1c_pointing_set_transforms)

            # ----
            transform_glows_psets(glows_pset)

            variables_to_merge = [
                exposure_factor_var_name,
                CoordNames.TIME.value,
                CoordNames.ELEVATION_L1C.value,
                CoordNames.AZIMUTH_L1C.value
            ]

            # does this merging make sense?
            merged_set = AddPsetData(l1c_pset.data, variables_to_merge).transform(glows_pset)

            for t in merged_pointing_set_transforms:
                merged_set = t.transform(merged_set)

            derived_psets.append(merged_set)

        sp_skymap = SpacecraftFrameProjection(SpiceFrame.ECLIPJ2000,
                                              [sp_var_name, exposure_factor_var_name]).project_rectangular(
            derived_psets, 4)
        sp_skymap = DivideOutWeights([sp_var_name], exposure_factor_var_name).transform(sp_skymap)

        sp_corrected_l2_map = SurvivalProbabilityCorrection(
            sp_skymap.data_1d[sp_var_name].values).transform(l2)

        return sp_corrected_l2_map
