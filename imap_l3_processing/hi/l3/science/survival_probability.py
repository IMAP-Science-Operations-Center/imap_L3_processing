import enum

import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry
from imap_processing.spice.time import TTJ2000_EPOCH
from spiceypy import spiceypy

from imap_l3_processing.hi.l3.models import HiL1cData, GlowsL3eData


class Sensor(enum.Enum):
    Hi45 = "45"
    Hi90 = "90"

    @staticmethod
    def get_sensor_angle(sensor_name):
        sensor_angles = {Sensor.Hi45.value: -45, Sensor.Hi90.value: 0}
        return sensor_angles[sensor_name]


class HiSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: HiL1cData, sensor: Sensor, glows_dataset: GlowsL3eData):
        super().__init__(xr.Dataset(), geometry.SpiceFrame.IMAP_DPS)
        glows_spin_bin_count = len(glows_dataset.spin_angle)
        survival_probabilities = np.empty(shape=(1, len(l1c_dataset.esa_energy_step), glows_spin_bin_count))
        for spin_angle_index in range(glows_spin_bin_count):
            survival_probabilities[0, :, spin_angle_index] = np.interp(
                np.log10(l1c_dataset.esa_energy_step),
                np.log10(glows_dataset.energy),
                glows_dataset.probability_of_survival[0, :, spin_angle_index], )
        survival_probabilities = np.repeat(survival_probabilities, 10, axis=2)

        azimuth_range = (0, 360)
        deg_spacing = 0.1

        half_bin_width = deg_spacing / 2

        sensor_angle = Sensor.get_sensor_angle(sensor.value)
        self.azimuths = np.arange(*azimuth_range, deg_spacing) + half_bin_width
        self.elevations = np.repeat(sensor_angle, 3600)
        self.az_el_points = np.column_stack([self.azimuths, self.elevations])

        self.num_points = l1c_dataset.exposure_times.shape[-1]
        self.spatial_coords = [CoordNames.AZIMUTH_L1C.value]

        # et_time = spiceypy.datetime2et(l1c_dataset.epoch)

        self.data = xr.Dataset({
            "survival_probability_times_exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                ],
                survival_probabilities * l1c_dataset.exposure_times,
            ),
            "exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                ],
                l1c_dataset.exposure_times,
            )
        },
            coords={
                CoordNames.TIME.value: np.array([l1c_dataset.epoch]).astype(np.datetime64) - TTJ2000_EPOCH,
                CoordNames.ENERGY.value: l1c_dataset.esa_energy_step,
                CoordNames.AZIMUTH_L1C.value: self.azimuths,
            }
        )


class HiSurvivalProbabilitySkyMap(RectangularSkyMap):
    def __init__(self, survival_probability_pointing_sets: list[HiSurvivalProbabilityPointingSet],
                 spacing_degree: float, spice_frame: geometry.SpiceFrame):
        super().__init__(spacing_degree, spice_frame)
        for sp_pset in survival_probability_pointing_sets:
            self.project_pset_values_to_map(sp_pset, ["survival_probability_times_exposure", "exposure"])

        self.data_1d = xr.Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure"]
        })
