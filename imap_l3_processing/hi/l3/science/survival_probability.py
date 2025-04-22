import numpy as np
import xarray as xr
from imap_l3_processing.hi.l3.models import HiL1cData, GlowsL3eData
from imap_l3_processing.hi.l3.utils import Sensor, SpinPhase
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry


class HiSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: HiL1cData, sensor: Sensor, spin_phase: SpinPhase, glows_dataset: GlowsL3eData,
                 energies: np.ndarray):
        super().__init__(xr.Dataset(), geometry.SpiceFrame.IMAP_DPS)
        glows_spin_bin_count = len(glows_dataset.spin_angle)
        survival_probabilities = np.empty(shape=(1, len(energies), glows_spin_bin_count))
        for spin_angle_index in range(glows_spin_bin_count):
            survival_probabilities[0, :, spin_angle_index] = np.interp(
                np.log10(energies),
                np.log10(glows_dataset.energy),
                glows_dataset.probability_of_survival[0, :, spin_angle_index], )
        survival_probabilities = np.repeat(survival_probabilities, 10, axis=2)

        num_spin_angle_bins = l1c_dataset.exposure_times.shape[-1]

        exposure_mask = np.full(num_spin_angle_bins, False)

        if spin_phase == SpinPhase.RamOnly:
            exposure_mask[0:900] = True
            exposure_mask[2700:3600] = True
        elif spin_phase == SpinPhase.AntiRamOnly:
            exposure_mask[900:2700] = True
        else:
            raise ValueError("Should not survival correct a full spin map!")

        exposure = l1c_dataset.exposure_times * exposure_mask

        spin_angle_range = (0, 360)
        deg_spacing = 360 / num_spin_angle_bins

        half_bin_width = deg_spacing / 2

        spin_angles = np.linspace(spin_angle_range[0], spin_angle_range[1], num_spin_angle_bins,
                                  endpoint=False) + half_bin_width

        sensor_angle = Sensor.get_sensor_angle(sensor.value)
        self.azimuths = spin_angles + 90
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
                survival_probabilities * exposure,
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
