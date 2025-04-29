from typing import Optional

import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry

from imap_l3_processing.hi.l3.models import HiL1cData, HiGlowsL3eData
from imap_l3_processing.hi.l3.utils import Sensor, SpinPhase


def interpolate_angular_data_to_nearest_neighbor(input_azimuths: np.array, glows_azimuths: np.array,
                                                 glows_data: np.array) -> np.array:
    expanded_az = np.concatenate([glows_azimuths - 360, glows_azimuths, glows_azimuths + 360])
    expanded_glows_data = np.concatenate([glows_data, glows_data, glows_data])
    sort = np.argsort(expanded_az)
    sorted_az = expanded_az[sort]
    sorted_data = expanded_glows_data[sort]
    bin_edges = sorted_az[:-1] + np.diff(sorted_az) / 2
    return sorted_data[np.digitize(input_azimuths, bin_edges, right=True)]


class HiSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: HiL1cData, sensor: Sensor, spin_phase: SpinPhase,
                 glows_dataset: Optional[HiGlowsL3eData],
                 energies: np.ndarray):
        super().__init__(xr.Dataset(), geometry.SpiceFrame.IMAP_DPS)
        num_spin_angle_bins = l1c_dataset.exposure_times.shape[-1]
        deg_spacing = 360 / num_spin_angle_bins
        half_bin_width = deg_spacing / 2
        spin_angles = np.linspace(0, 360, num_spin_angle_bins,
                                  endpoint=False) + half_bin_width
        self.azimuths = np.mod(spin_angles + 90, 360)

        if glows_dataset is not None:
            glows_spin_bin_count = len(glows_dataset.spin_angle)
            sp_interpolated_to_hi_energies = np.empty(shape=(len(energies), glows_spin_bin_count))
            for spin_angle_index in range(glows_spin_bin_count):
                sp_interpolated_to_hi_energies[:, spin_angle_index] = np.interp(
                    np.log10(energies),
                    np.log10(glows_dataset.energy),
                    glows_dataset.probability_of_survival[0, :, spin_angle_index], )

            sp_interpolated_to_pset_angles = np.zeros((1, len(energies), 3600))
            for e_index in range(len(energies)):
                sp_interpolated_to_pset_angles[0, e_index] = interpolate_angular_data_to_nearest_neighbor(
                    self.azimuths, glows_dataset.spin_angle, sp_interpolated_to_hi_energies[e_index])
        else:
            sp_interpolated_to_pset_angles = np.full((1, len(energies), 3600), np.nan)

        exposure_mask = np.full(num_spin_angle_bins, False)

        assert num_spin_angle_bins == 3600, "unexpected number of spin angles"
        if spin_phase == SpinPhase.RamOnly:
            exposure_mask[0:900] = True
            exposure_mask[2700:3600] = True
        elif spin_phase == SpinPhase.AntiRamOnly:
            exposure_mask[900:2700] = True
        else:
            raise ValueError("Should not survival correct a full spin map!")

        exposure = l1c_dataset.exposure_times * exposure_mask

        sensor_angle = Sensor.get_sensor_angle(sensor.value)
        self.elevations = np.repeat(sensor_angle, num_spin_angle_bins)
        self.az_el_points = np.column_stack([self.azimuths, self.elevations])

        self.num_points = num_spin_angle_bins
        self.spatial_coords = [CoordNames.AZIMUTH_L1C.value]

        self.data = xr.Dataset({
            "survival_probability_times_exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                ],
                sp_interpolated_to_pset_angles * exposure,
            ),
            "exposure": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY.value,
                    CoordNames.AZIMUTH_L1C.value,
                ],
                exposure,
            )
        },
            coords={
                CoordNames.TIME.value: l1c_dataset.epoch_j2000,
                CoordNames.ENERGY.value: l1c_dataset.esa_energy_step,
                CoordNames.AZIMUTH_L1C.value: self.azimuths,
            },
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
