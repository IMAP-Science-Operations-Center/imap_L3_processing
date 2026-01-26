import numpy as np
import xarray as xr
from imap_processing.ena_maps.ena_maps import RectangularSkyMap, PointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import apply_compton_getting_correction, \
    add_spacecraft_velocity_to_pset, calculate_ram_mask
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.maps.map_descriptors import Sensor, SpinPhase
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet


def interpolate_angular_data_to_nearest_neighbor(input_azimuths: np.array, glows_azimuths: np.array,
                                                 glows_data: np.array) -> np.array:
    expanded_az = np.concatenate([glows_azimuths - 360, glows_azimuths, glows_azimuths + 360])
    expanded_glows_data = np.concatenate([glows_data, glows_data, glows_data], axis=1)
    sort = np.argsort(expanded_az)
    sorted_az = expanded_az[sort]
    sorted_data = expanded_glows_data[:, sort]
    bin_edges = sorted_az[:-1] + np.diff(sorted_az) / 2
    return sorted_data[:, np.digitize(input_azimuths, bin_edges, right=True)]


class RectangularSurvivalProbabilityPointingSet(PointingSet):
    def __init__(self, l1c_dataset: InputRectangularPointingSet, sensor: Sensor, spin_phase: SpinPhase,
                 glows_dataset: GlowsL3eRectangularMapInputData, energies: np.ndarray, cg_corrected: bool = False):
        num_spin_angle_bins = l1c_dataset.exposure_times.shape[-1]

        deg_spacing = 360 / num_spin_angle_bins
        half_bin_width = deg_spacing / 2

        self.azimuths = np.linspace(0, 360, num_spin_angle_bins,
                                    endpoint=False) + half_bin_width

        sensor_angle = Sensor.get_sensor_angle(sensor)
        self.elevations = np.repeat(sensor_angle, num_spin_angle_bins)

        hae_az_el_points = xr.DataArray(
            np.column_stack([l1c_dataset.hae_longitude[0], l1c_dataset.hae_latitude[0]]),
            dims=[CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
        )

        self.spatial_coords = (CoordNames.AZIMUTH_L1C.value,)

        if l1c_dataset.epoch_delta is not None:
            repointing_midpoint = l1c_dataset.epoch_j2000 + l1c_dataset.epoch_delta / 2
        elif l1c_dataset.pointing_start_met is not None and l1c_dataset.pointing_end_met is not None:
            pointing_duration = l1c_dataset.pointing_end_met - l1c_dataset.pointing_start_met
            repointing_midpoint = l1c_dataset.epoch_j2000 + (1e9 * pointing_duration / 2)

        initial_dataset = xr.Dataset({},
                                     coords={
                                         CoordNames.TIME.value: l1c_dataset.epoch_j2000,
                                         CoordNames.ENERGY_ULTRA_L1C.value: l1c_dataset.esa_energy_step,
                                         CoordNames.AZIMUTH_L1C.value: self.azimuths,
                                     })

        assert num_spin_angle_bins == 3600, "unexpected number of spin angles"

        initial_dataset['epoch'] = l1c_dataset.epoch_j2000
        initial_dataset['epoch_delta'] = l1c_dataset.epoch_delta
        initial_dataset['hae_longitude'] = xr.DataArray(
            l1c_dataset.hae_longitude,
            dims=[CoordNames.TIME.value, CoordNames.AZIMUTH_L1C.value],
        )
        initial_dataset['hae_latitude'] = xr.DataArray(
            l1c_dataset.hae_latitude,
            dims=[CoordNames.TIME.value, CoordNames.AZIMUTH_L1C.value],
        )
        initial_dataset['pointing_start_met'] = l1c_dataset.pointing_start_met
        initial_dataset['pointing_end_met'] = l1c_dataset.pointing_end_met

        if sensor == Sensor.Hi90 or sensor == Sensor.Hi45:
            initial_dataset.attrs['Logical_source'] = 'imap_hi'
        elif sensor in (Sensor.Lo90, Sensor.Lo):
            initial_dataset.attrs['Logical_source'] = 'imap_lo'
        dataset = add_spacecraft_velocity_to_pset(initial_dataset)

        if cg_corrected:
            energy_in_ev = energies * 1000

            dataset = apply_compton_getting_correction(
                dataset,
                xr.DataArray(energy_in_ev, dims=[CoordNames.ENERGY_ULTRA_L1C.value])
            )
            self.az_el_points = xr.DataArray(
                np.stack([dataset['hae_longitude'].values[0], dataset['hae_latitude'].values[0]], axis=2),
                dims=[CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
            )

            exposure = np.full_like(l1c_dataset.exposure_times, np.nan)
            best_match_energies = np.full_like(l1c_dataset.exposure_times, np.nan, dtype=np.float64)
            for cg_energy_index, cg_energy in np.ndenumerate(dataset['energy_sc'].values[0]):
                best_guess = np.inf
                best_guess_index = -1
                for e_i, energy in enumerate(energy_in_ev):
                    guess = np.abs(energy - cg_energy)
                    if guess < best_guess:
                        best_guess = guess
                        best_guess_index = e_i
                    if guess > best_guess:
                        break
                exposure[0, cg_energy_index[0], cg_energy_index[1]] = l1c_dataset.exposure_times[
                    0, best_guess_index, cg_energy_index[1]]
                if sensor in [Sensor.Lo, Sensor.Lo90]:
                    best_match_energies[0, cg_energy_index[0], cg_energy_index[1]] = cg_energy / 1000
                else:
                    best_match_energies[0, cg_energy_index[0], cg_energy_index[1]] = energies[best_guess_index]

        else:
            self.az_el_points = hae_az_el_points
            exposure = l1c_dataset.exposure_times

        dataset = calculate_ram_mask(dataset)

        if spin_phase == SpinPhase.RamOnly:
            dataset["directional_mask"] = dataset["ram_mask"]
        else:
            dataset["directional_mask"] = ~dataset["ram_mask"]

        if glows_dataset is not None:
            if cg_corrected:
                sp_interpolated_to_pset_angles = interpolate_angular_data_to_nearest_neighbor(
                    self.azimuths, glows_dataset.spin_angle, glows_dataset.probability_of_survival[0])
                log_sc_frame_energies = np.log10(best_match_energies[0])

                sp_final = np.empty((1, len(energies), len(self.azimuths)))
                for spin_angle_index in range(len(self.azimuths)):
                    sp_final[0, :, spin_angle_index] = np.interp(
                        log_sc_frame_energies[:, spin_angle_index],
                        np.log10(glows_dataset.energy),
                        sp_interpolated_to_pset_angles[:, spin_angle_index]
                    )
            else:
                glows_spin_bin_count = len(glows_dataset.spin_angle)
                sp_interpolated_to_hi_energies = np.empty(shape=(len(energies), glows_spin_bin_count))
                for spin_angle_index in range(glows_spin_bin_count):
                    sp_interpolated_to_hi_energies[:, spin_angle_index] = np.interp(
                        np.log10(energies),
                        np.log10(glows_dataset.energy),
                        glows_dataset.probability_of_survival[0, :, spin_angle_index])

                sp_interpolated_to_pset_angles = np.zeros((1, len(energies), 3600))
                sp_interpolated_to_pset_angles[0] = interpolate_angular_data_to_nearest_neighbor(
                    self.azimuths, glows_dataset.spin_angle, sp_interpolated_to_hi_energies)
                sp_final = sp_interpolated_to_pset_angles
        else:
            sp_final = np.full((1, len(energies), 3600), np.nan)

        dataset["survival_probability_times_exposure"] = xr.DataArray(
            sp_final * exposure,
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

        frame = SpiceFrame.IMAP_HAE
        super().__init__(dataset, frame)


class RectangularSurvivalProbabilitySkyMap(RectangularSkyMap):
    def __init__(self, survival_probability_pointing_sets: list[RectangularSurvivalProbabilityPointingSet],
                 spacing_degree: float, spice_frame: SpiceFrame):
        super().__init__(spacing_degree, spice_frame)
        for sp_pset in survival_probability_pointing_sets:
            value_keys = ["survival_probability_times_exposure", "exposure"]
            self.project_pset_values_to_map(sp_pset, value_keys, pset_valid_mask=sp_pset.data["directional_mask"])

        self.data_1d = xr.Dataset({
            "exposure_weighted_survival_probabilities": self.data_1d["survival_probability_times_exposure"] /
                                                        self.data_1d["exposure"]
        })
