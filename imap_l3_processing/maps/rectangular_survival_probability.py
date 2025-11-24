import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import apply_compton_getting_correction
from imap_processing.spice.geometry import SpiceFrame, frame_transform_az_el
from imap_processing.spice.time import ttj2000ns_to_et

from imap_l3_processing.maps.map_descriptors import Sensor, SpinPhase
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet


def interpolate_angular_data_to_nearest_neighbor(input_azimuths: np.array, glows_azimuths: np.array,
                                                 glows_data: np.array) -> np.array:
    expanded_az = np.concatenate([glows_azimuths - 360, glows_azimuths, glows_azimuths + 360])
    expanded_glows_data = np.concatenate([glows_data, glows_data, glows_data])
    sort = np.argsort(expanded_az)
    sorted_az = expanded_az[sort]
    sorted_data = expanded_glows_data[sort]
    bin_edges = sorted_az[:-1] + np.diff(sorted_az) / 2
    return sorted_data[np.digitize(input_azimuths, bin_edges, right=True)]


class RectangularSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: InputRectangularPointingSet, sensor: Sensor, spin_phase: SpinPhase,
                 glows_dataset: GlowsL3eRectangularMapInputData, energies: np.ndarray, cg_corrected: bool = False):
        num_spin_angle_bins = l1c_dataset.exposure_times.shape[-1]
        deg_spacing = 360 / num_spin_angle_bins
        half_bin_width = deg_spacing / 2
        spin_angles = np.linspace(0, 360, num_spin_angle_bins,
                                  endpoint=False) + half_bin_width
        self.azimuths = np.mod(spin_angles + 90, 360)

        sensor_angle = Sensor.get_sensor_angle(sensor)
        self.elevations = np.repeat(sensor_angle, num_spin_angle_bins)

        spacecraft_az_el_points = xr.DataArray(
            np.column_stack([self.azimuths, self.elevations]),
            dims=[CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
        )

        self.spatial_coords = (CoordNames.AZIMUTH_L1C.value,)

        repointing_midpoint = l1c_dataset.epoch_j2000 + l1c_dataset.epoch_delta / 2

        initial_dataset = xr.Dataset({},
                                     coords={
                                         CoordNames.TIME.value: l1c_dataset.epoch_j2000,
                                         CoordNames.ENERGY_ULTRA_L1C.value: l1c_dataset.esa_energy_step,
                                         CoordNames.AZIMUTH_L1C.value: self.azimuths,
                                     })

        exposure_mask = np.full(num_spin_angle_bins, False)

        assert num_spin_angle_bins == 3600, "unexpected number of spin angles"
        if spin_phase == SpinPhase.RamOnly:
            exposure_mask[0:900] = True
            exposure_mask[2700:3600] = True
        elif spin_phase == SpinPhase.AntiRamOnly:
            exposure_mask[900:2700] = True
        else:
            raise ValueError("Should not survival correct a full spin map!")

        if cg_corrected:
            et = ttj2000ns_to_et(repointing_midpoint)

            hae_az_el = frame_transform_az_el(
                et=et,
                az_el=spacecraft_az_el_points.values,
                from_frame=SpiceFrame.IMAP_DPS,
                to_frame=SpiceFrame.IMAP_HAE,
                degrees=True
            )

            initial_dataset['epoch'] = l1c_dataset.epoch_j2000
            initial_dataset['epoch_delta'] = l1c_dataset.epoch_delta
            initial_dataset['hae_longitude'] = hae_az_el[:, 0]
            initial_dataset['hae_latitude'] = hae_az_el[:, 1]

            energy_in_eV = energies * 1000
            dataset = apply_compton_getting_correction(initial_dataset, xr.DataArray(energy_in_eV))
            self.az_el_points = xr.DataArray(
                np.stack([dataset['hae_longitude'].values, dataset['hae_latitude'].values], axis=2),
                dims=[CoordNames.ENERGY_L2.value, CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
            )

            exposures = np.full_like(l1c_dataset.exposure_times, np.nan)
            for cg_energy_index, cg_energy in np.ndenumerate(dataset['energy_sc'].values):
                best_guess = np.inf
                best_guess_index = -1
                for e_i, energy in enumerate(energies):
                    guess = np.abs(energy - cg_energy)
                    if guess < best_guess:
                        best_guess = guess
                        best_guess_index = e_i
                    if guess > best_guess:
                        break
                exposures[0, cg_energy_index, :] = l1c_dataset.exposure_times[0, best_guess_index, :]

            exposure = exposures * exposure_mask

        else:
            dataset = initial_dataset
            self.az_el_points = spacecraft_az_el_points
            exposure = l1c_dataset.exposure_times * exposure_mask

        if glows_dataset is not None:
            glows_spin_bin_count = len(glows_dataset.spin_angle)
            sp_interpolated_to_hi_energies = np.empty(shape=(len(energies), glows_spin_bin_count))
            for spin_angle_index in range(glows_spin_bin_count):
                energies_to_interpolate = dataset['energy_sc'].values if cg_corrected else energies
                sp_interpolated_to_hi_energies[:, spin_angle_index] = np.interp(
                    np.log10(energies_to_interpolate),
                    np.log10(glows_dataset.energy),
                    glows_dataset.probability_of_survival[0, :, spin_angle_index], )

            sp_interpolated_to_pset_angles = np.zeros((1, len(energies), 3600))
            for e_index in range(len(energies)):
                sp_interpolated_to_pset_angles[0, e_index] = interpolate_angular_data_to_nearest_neighbor(
                    self.azimuths, glows_dataset.spin_angle, sp_interpolated_to_hi_energies[e_index])
        else:
            sp_interpolated_to_pset_angles = np.full((1, len(energies), 3600), np.nan)

        dataset["survival_probability_times_exposure"] = xr.DataArray(
            sp_interpolated_to_pset_angles * exposure,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ]
        )
        dataset["exposure"] = xr.DataArray(
            exposure,
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_ULTRA_L1C.value,
                CoordNames.AZIMUTH_L1C.value,
            ],
        )
        dataset["epoch"] = repointing_midpoint

        super().__init__(dataset, SpiceFrame.IMAP_DPS)


class RectangularSurvivalProbabilitySkyMap(RectangularSkyMap):
    def __init__(self, survival_probability_pointing_sets: list[RectangularSurvivalProbabilityPointingSet],
                 spacing_degree: float, spice_frame: SpiceFrame):
        super().__init__(spacing_degree, spice_frame)
        for sp_pset in survival_probability_pointing_sets:
            self.project_pset_values_to_map(sp_pset, ["survival_probability_times_exposure", "exposure"])

        self.data_1d = xr.Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure"]
        })
