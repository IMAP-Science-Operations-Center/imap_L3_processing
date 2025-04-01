import enum

import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularPointingSet, RectangularSkyMap, AbstractSkyMap
from imap_processing.ena_maps.utils.coordinates import CoordNames


class Sensor(enum.Enum):
    Hi45 = "hi-45"
    Hi90 = "hi-90"

    @staticmethod
    def get_sensor_angle(sensor_name):
        sensor_angles = {Sensor.Hi45.value: -45, Sensor.Hi90.value: 0}
        return sensor_angles[sensor_name]


class HiSurvivalProbabilityPointingSet(RectangularPointingSet):
    def __init__(self, l1c_dataset: xr.Dataset, sensor: Sensor, glows_dataset: xr.Dataset):
        num_epochs = len(l1c_dataset["epoch"].values)
        num_energies = len(l1c_dataset["esa_energy_step"].values)

        azimuth_range = (0, 360)
        elevation_range = (-90, 90)
        deg_spacing = 0.1

        half_bin_width = deg_spacing / 2
        elevations = np.arange(*elevation_range, deg_spacing) + half_bin_width
        azimuths = np.arange(*azimuth_range, deg_spacing) + half_bin_width

        sensor_angle = Sensor.get_sensor_angle(sensor.value)

        elevation_bin_edges = np.concatenate([elevations - half_bin_width, [elevations[-1] + half_bin_width]])
        elevation_bin_for_sensor_angle = np.digitize(sensor_angle, elevation_bin_edges)

        survival_probabilities = np.full((num_epochs, num_energies, len(azimuths), len(elevations)), 0)
        survival_probablities_raw = np.repeat(glows_dataset['probability_of_survival'].values, axis=2, repeats=10)
        for spin_angle_index in range(len(glows_dataset['spin_angle_bin'].values) * 10):
            survival_probabilities[0, :, spin_angle_index, elevation_bin_for_sensor_angle] = np.interp(
                np.log10(l1c_dataset['esa_energy_step'].values),
                np.log10(glows_dataset['energy'].values),
                survival_probablities_raw[0, :, spin_angle_index], )
        exposure = np.full(
            (num_epochs, num_energies, len(azimuths), len(elevations)),
            fill_value=0, dtype=np.float64)

        exposure[:, :, :, elevation_bin_for_sensor_angle] = l1c_dataset['exposure_times'].values

        survival_probabilities_by_exposure = survival_probabilities * exposure

        self.data = xr.Dataset({
            "survival_probability_times_exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                    CoordNames.ELEVATION_L1C.value,
                ],
                survival_probabilities_by_exposure,
            ),
            "exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                    CoordNames.ELEVATION_L1C.value,
                ],
                exposure,
            )
        },
            coords={
                CoordNames.TIME.value: l1c_dataset["epoch"].values,
                CoordNames.ENERGY.value: l1c_dataset["esa_energy_step"].values,
                CoordNames.AZIMUTH_L1C.value: azimuths,
                CoordNames.ELEVATION_L1C.value: elevations,
            }
        )
        super().__init__(self.data)
