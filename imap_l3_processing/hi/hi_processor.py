import numpy as np
from imap_data_access import upload
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies import HiL3CombinedMapDependencies
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.hi.l3.models import combine_maps, \
    HiIntensityMapData, HiSpectralMapData, HiL3IntensityDataProduct, HiL3SpectralIndexDataProduct, \
    calculate_datetime_weighted_average
from imap_l3_processing.hi.l3.science.spectral_fit import spectral_fit
from imap_l3_processing.hi.l3.science.survival_probability import HiSurvivalProbabilityPointingSet, \
    HiSurvivalProbabilitySkyMap
from imap_l3_processing.hi.l3.utils import parse_map_descriptor, MapQuantity, MapDescriptorParts, SurvivalCorrection, \
    SpinPhase, Duration, Sensor
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data, combine_glows_l3e_with_l1c_pointing


class HiProcessor(Processor):
    def process(self):
        set_of_parent_file_names = set(self.get_parent_file_names())

        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                hi_l3_spectral_fit_dependencies = HiL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                map = self.process_spectral_fit_index(hi_l3_spectral_fit_dependencies)
                data_product = HiL3SpectralIndexDataProduct(
                    data=map,
                    input_metadata=self.input_metadata,
                )
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.RamOnly | SpinPhase.AntiRamOnly,
                                    duration=Duration.SixMonths):
                hi_l3_survival_probabilities_dependencies = HiL3SurvivalDependencies.fetch_dependencies(
                    self.dependencies)
                data_product = HiL3IntensityDataProduct(
                    data=self.process_survival_probabilities(hi_l3_survival_probabilities_dependencies),
                    input_metadata=self.input_metadata,
                )
                set_of_parent_file_names.update(
                    p.name for p in hi_l3_survival_probabilities_dependencies.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.FullSpin,
                                    duration=Duration.SixMonths):
                hi_l3_full_spin_dependencies = HiL3SingleSensorFullSpinDependencies.fetch_dependencies(
                    self.dependencies)
                combined_map = self.process_full_spin_single_sensor(hi_l3_full_spin_dependencies)
                data_product = HiL3IntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata
                )
                set_of_parent_file_names.update(p.name for p in hi_l3_full_spin_dependencies.dependency_file_paths)
            case MapDescriptorParts(sensor=Sensor.Combined,
                                    spin_phase=SpinPhase.FullSpin,
                                    duration=Duration.OneYear,
                                    ):
                downloaded_deps = HiL3CombinedMapDependencies.fetch_dependencies(self.dependencies)
                combined_map = combine_maps(downloaded_deps.maps)
                data_product = HiL3IntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata,
                )
            case None:
                raise ValueError(f"Could not parse descriptor {self.input_metadata.descriptor}")
            case _:
                raise NotImplementedError(self.input_metadata.descriptor)

        data_product.parent_file_names = sorted(set_of_parent_file_names)

        cdf_path = save_data(data_product)
        upload(cdf_path)

    def process_full_spin_single_sensor(self,
                                        hi_l3_full_spin_dependencies: HiL3SingleSensorFullSpinDependencies) -> HiIntensityMapData:
        ram_data_product = self.process_survival_probabilities(hi_l3_full_spin_dependencies.ram_dependencies)
        antiram_data_product = self.process_survival_probabilities(hi_l3_full_spin_dependencies.antiram_dependencies)

        return combine_maps([ram_data_product, antiram_data_product])

    def process_spectral_fit_index(self,
                                   hi_l3_spectral_fit_dependencies: HiL3SpectralFitDependencies) -> HiSpectralMapData:
        input_data = hi_l3_spectral_fit_dependencies.hi_l3_data
        hi_l3_data = input_data

        epochs = hi_l3_data.epoch
        energy = hi_l3_data.energy
        fluxes = hi_l3_data.ena_intensity
        lons = hi_l3_data.longitude
        lats = hi_l3_data.latitude
        variances = np.square(hi_l3_data.ena_intensity_stat_unc)

        gammas, errors = spectral_fit(len(epochs), len(lons), len(lats), fluxes, variances, energy)

        min_energy = hi_l3_data.energy[0] - hi_l3_data.energy_delta_minus[0]
        max_energy = hi_l3_data.energy[-1] + hi_l3_data.energy_delta_plus[-1]
        mean_energy = np.sqrt(min_energy * max_energy)

        new_energy_label = f"{min_energy} - {max_energy} keV"

        mean_obs_date = calculate_datetime_weighted_average(input_data.obs_date, weights=input_data.exposure_factor,
                                                            axis=1, keepdims=True)
        mean_obs_date_range = np.ma.average(input_data.obs_date_range, weights=input_data.exposure_factor, axis=1,
                                            keepdims=True)
        total_exposure_factor = np.sum(input_data.exposure_factor, axis=1, keepdims=True)

        return HiSpectralMapData(
            ena_spectral_index_stat_unc=errors[:, np.newaxis, :, :],
            ena_spectral_index=gammas[:, np.newaxis, :, :],
            epoch=input_data.epoch,
            epoch_delta=input_data.epoch_delta,
            energy=np.array([mean_energy]),
            energy_delta_plus=np.array([max_energy - mean_energy]),
            energy_delta_minus=np.array([mean_energy - min_energy]),
            energy_label=np.array([new_energy_label]),
            latitude=input_data.latitude,
            latitude_delta=input_data.latitude_delta,
            latitude_label=input_data.latitude_label,
            longitude=input_data.longitude,
            longitude_delta=input_data.longitude_delta,
            longitude_label=input_data.longitude_label,
            exposure_factor=total_exposure_factor,
            obs_date=mean_obs_date,
            obs_date_range=mean_obs_date_range,
            solid_angle=input_data.solid_angle,
        )

    def process_survival_probabilities(self, hi_survival_probabilities_dependencies: HiL3SurvivalDependencies) \
            -> HiIntensityMapData:
        l2_descriptor_parts = hi_survival_probabilities_dependencies.l2_map_descriptor_parts

        combined_glows_hi = combine_glows_l3e_with_l1c_pointing(hi_survival_probabilities_dependencies.glows_l3e_data,
                                                                hi_survival_probabilities_dependencies.hi_l1c_data)
        pointing_sets = []
        for hi_l1c, glows_l3e in combined_glows_hi:
            pointing_sets.append(HiSurvivalProbabilityPointingSet(
                hi_l1c, l2_descriptor_parts.sensor, l2_descriptor_parts.spin_phase, glows_l3e,
                hi_survival_probabilities_dependencies.l2_data.energy))
        assert len(pointing_sets) > 0

        hi_survival_sky_map = HiSurvivalProbabilitySkyMap(pointing_sets, int(l2_descriptor_parts.grid),
                                                          SpiceFrame.ECLIPJ2000)

        survival_dataset = hi_survival_sky_map.to_dataset()

        input_data = hi_survival_probabilities_dependencies.l2_data
        survival_probabilities = survival_dataset["exposure_weighted_survival_probabilities"].values

        survival_corrected_intensity = input_data.ena_intensity / survival_probabilities
        corrected_stat_unc = input_data.ena_intensity_stat_unc / survival_probabilities
        corrected_sys_unc = input_data.ena_intensity_sys_err / survival_probabilities

        return HiIntensityMapData(
            ena_intensity_stat_unc=corrected_stat_unc,
            ena_intensity_sys_err=corrected_sys_unc,
            ena_intensity=survival_corrected_intensity,
            epoch=input_data.epoch,
            epoch_delta=input_data.epoch_delta,
            energy=input_data.energy,
            energy_delta_plus=input_data.energy_delta_plus,
            energy_delta_minus=input_data.energy_delta_minus,
            energy_label=input_data.energy_label,
            latitude=input_data.latitude,
            latitude_delta=input_data.latitude_delta,
            latitude_label=input_data.latitude_label,
            longitude=input_data.longitude,
            longitude_delta=input_data.longitude_delta,
            longitude_label=input_data.longitude_label,
            exposure_factor=input_data.exposure_factor,
            obs_date=input_data.obs_date,
            obs_date_range=input_data.obs_date_range,
            solid_angle=input_data.solid_angle,
        )
