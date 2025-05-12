import numpy as np
from imap_data_access import upload
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies import HiL3CombinedMapDependencies
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.hi.l3.science.spectral_fit import spectral_fit
from imap_l3_processing.hi.l3.science.survival_probability import HiSurvivalProbabilityPointingSet, \
    HiSurvivalProbabilitySkyMap
from imap_l3_processing.hi.l3.utils import parse_map_descriptor, MapQuantity, MapDescriptorParts, SurvivalCorrection, \
    SpinPhase, Duration, Sensor
from imap_l3_processing.map_models import RectangularSpectralIndexDataProduct, RectangularSpectralIndexMapData, \
    SpectralIndexMapData, RectangularIntensityMapData, IntensityMapData, RectangularIntensityDataProduct, \
    combine_rectangular_intensity_map_data, calculate_datetime_weighted_average
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data, combine_glows_l3e_with_l1c_pointing


class HiProcessor(Processor):
    def process(self):
        set_of_parent_file_names = set(self.get_parent_file_names())

        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                hi_l3_spectral_fit_dependencies = HiL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                map_data = self.process_spectral_fit_index(hi_l3_spectral_fit_dependencies)
                data_product = RectangularSpectralIndexDataProduct(
                    data=map_data,
                    input_metadata=self.input_metadata,
                )
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.RamOnly | SpinPhase.AntiRamOnly,
                                    duration=Duration.SixMonths):
                hi_l3_survival_probabilities_dependencies = HiL3SurvivalDependencies.fetch_dependencies(
                    self.dependencies)
                data_product = RectangularIntensityDataProduct(
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
                data_product = RectangularIntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata
                )
                set_of_parent_file_names.update(p.name for p in hi_l3_full_spin_dependencies.dependency_file_paths)
            case MapDescriptorParts(sensor=Sensor.Combined,
                                    spin_phase=SpinPhase.FullSpin,
                                    duration=Duration.OneYear,
                                    ):
                downloaded_deps = HiL3CombinedMapDependencies.fetch_dependencies(self.dependencies)
                combined_map = combine_rectangular_intensity_map_data(downloaded_deps.maps)
                data_product = RectangularIntensityDataProduct(
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

    def process_full_spin_single_sensor(self, hi_l3_full_spin_dependencies: HiL3SingleSensorFullSpinDependencies) \
            -> RectangularIntensityMapData:
        ram_data_product = self.process_survival_probabilities(hi_l3_full_spin_dependencies.ram_dependencies)
        antiram_data_product = self.process_survival_probabilities(hi_l3_full_spin_dependencies.antiram_dependencies)

        return combine_rectangular_intensity_map_data([ram_data_product, antiram_data_product])

    def process_spectral_fit_index(self, hi_l3_spectral_fit_dependencies: HiL3SpectralFitDependencies) \
            -> RectangularSpectralIndexMapData:
        input_data = hi_l3_spectral_fit_dependencies.hi_l3_data
        intensity_data = input_data.intensity_map_data

        energy = intensity_data.energy
        fluxes = intensity_data.ena_intensity
        variances = np.square(intensity_data.ena_intensity_stat_unc)

        gammas, errors = spectral_fit(fluxes, variances, energy)

        min_energy = intensity_data.energy[0] - intensity_data.energy_delta_minus[0]
        max_energy = intensity_data.energy[-1] + intensity_data.energy_delta_plus[-1]
        mean_energy = np.sqrt(min_energy * max_energy)

        new_energy_label = f"{min_energy} - {max_energy} keV"

        mean_obs_date = calculate_datetime_weighted_average(intensity_data.obs_date,
                                                            weights=intensity_data.exposure_factor,
                                                            axis=1, keepdims=True)
        mean_obs_date_range = np.ma.average(intensity_data.obs_date_range, weights=intensity_data.exposure_factor,
                                            axis=1,
                                            keepdims=True)
        total_exposure_factor = np.sum(intensity_data.exposure_factor, axis=1, keepdims=True)
        return RectangularSpectralIndexMapData(
            spectral_index_map_data=SpectralIndexMapData(
                ena_spectral_index_stat_unc=errors,
                ena_spectral_index=gammas,
                epoch=intensity_data.epoch,
                epoch_delta=intensity_data.epoch_delta,
                energy=np.array([mean_energy]),
                energy_delta_plus=np.array([max_energy - mean_energy]),
                energy_delta_minus=np.array([mean_energy - min_energy]),
                energy_label=np.array([new_energy_label]),
                latitude=intensity_data.latitude,
                longitude=intensity_data.longitude,
                exposure_factor=total_exposure_factor,
                obs_date=mean_obs_date,
                obs_date_range=mean_obs_date_range,
                solid_angle=intensity_data.solid_angle,
            ),
            coords=input_data.coords
        )

    def process_survival_probabilities(self, hi_survival_probabilities_dependencies: HiL3SurvivalDependencies) \
            -> RectangularIntensityMapData:
        l2_descriptor_parts = hi_survival_probabilities_dependencies.l2_map_descriptor_parts

        combined_glows_hi = combine_glows_l3e_with_l1c_pointing(hi_survival_probabilities_dependencies.glows_l3e_data,
                                                                hi_survival_probabilities_dependencies.hi_l1c_data)
        pointing_sets = []
        input_data = hi_survival_probabilities_dependencies.l2_data.intensity_map_data

        for hi_l1c, glows_l3e in combined_glows_hi:
            pointing_sets.append(HiSurvivalProbabilityPointingSet(
                hi_l1c, l2_descriptor_parts.sensor, l2_descriptor_parts.spin_phase, glows_l3e,
                input_data.energy))
        assert len(pointing_sets) > 0

        hi_survival_sky_map = HiSurvivalProbabilitySkyMap(pointing_sets, int(l2_descriptor_parts.grid),
                                                          SpiceFrame.ECLIPJ2000)

        survival_dataset = hi_survival_sky_map.to_dataset()

        input_data = hi_survival_probabilities_dependencies.l2_data.intensity_map_data
        survival_probabilities = survival_dataset["exposure_weighted_survival_probabilities"].values

        survival_corrected_intensity = input_data.ena_intensity / survival_probabilities
        corrected_stat_unc = input_data.ena_intensity_stat_unc / survival_probabilities
        corrected_sys_unc = input_data.ena_intensity_sys_err / survival_probabilities

        return RectangularIntensityMapData(
            intensity_map_data=IntensityMapData(
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
                longitude=input_data.longitude,
                exposure_factor=input_data.exposure_factor,
                obs_date=input_data.obs_date,
                obs_date_range=input_data.obs_date_range,
                solid_angle=input_data.solid_angle,
            ),
            coords=hi_survival_probabilities_dependencies.l2_data.coords
        )
