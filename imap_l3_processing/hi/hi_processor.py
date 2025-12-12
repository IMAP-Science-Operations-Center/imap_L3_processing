from pathlib import Path

from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.hi.hi_combined_sensor_dependencies import HiL3CombinedMapDependencies
from imap_l3_processing.hi.hi_spectral_fit_dependencies import HiSpectralIndexDependencies
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.maps.map_combination import UncertaintyWeightedCombination, UnweightedCombination, \
    CombinationStrategy, ExposureWeightedCombination
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapQuantity, MapDescriptorParts, \
    SurvivalCorrection, \
    SpinPhase, Sensor, ReferenceFrame
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularSpectralIndexMapData, \
    RectangularIntensityMapData, RectangularIntensityDataProduct
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.spectral_fit import fit_spectral_index_map
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from imap_l3_processing.models import Instrument
from imap_l3_processing.utils import save_data


class HiProcessor(MapProcessor):
    def process(self, spice_frame_name: SpiceFrame = SpiceFrame.ECLIPJ2000) -> list[Path]:
        set_of_parent_file_names = set(self.get_parent_file_names())

        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                hi_l3_spectral_fit_dependencies = HiSpectralIndexDependencies.fetch_dependencies(self.dependencies)
                map_data = self.process_spectral_fit_index(hi_l3_spectral_fit_dependencies)
                data_product = RectangularSpectralIndexDataProduct(
                    data=map_data,
                    input_metadata=self.input_metadata,
                )
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.RamOnly | SpinPhase.AntiRamOnly):
                hi_l3_survival_probabilities_dependencies = HiLoL3SurvivalDependencies.fetch_dependencies(
                    self.dependencies, Instrument.IMAP_HI)
                data_product = RectangularIntensityDataProduct(
                    data=process_survival_probabilities(hi_l3_survival_probabilities_dependencies, spice_frame_name),
                    input_metadata=self.input_metadata,
                )
                set_of_parent_file_names.update(
                    p.name for p in hi_l3_survival_probabilities_dependencies.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.FullSpin,
                                    reference_frame=ReferenceFrame.Spacecraft):
                hi_l3_full_spin_dependencies = HiL3SingleSensorFullSpinDependencies.fetch_dependencies(
                    self.dependencies)
                combination_strategy = UnweightedCombination()
                combined_map = self.process_full_spin_single_sensor(hi_l3_full_spin_dependencies, spice_frame_name,
                                                                    combination_strategy)
                data_product = RectangularIntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata
                )
                set_of_parent_file_names.update(p.name for p in hi_l3_full_spin_dependencies.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.FullSpin,
                                    reference_frame=ReferenceFrame.Heliospheric):

                hi_l3_full_spin_cg_corrected_dependencies = HiL3SingleSensorFullSpinDependencies.fetch_dependencies(
                    self.dependencies)

                combination_strategy = UncertaintyWeightedCombination()
                combined_map = self.process_full_spin_single_sensor(hi_l3_full_spin_cg_corrected_dependencies,
                                                                    spice_frame_name, combination_strategy)
                data_product = RectangularIntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata
                )
            case MapDescriptorParts(sensor=Sensor.HiCombined):
                dependencies = HiL3CombinedMapDependencies.fetch_dependencies(self.dependencies)
                combined_map_data = ExposureWeightedCombination().combine_rectangular_intensity_map_data(
                    dependencies.maps)

                data_product = RectangularIntensityDataProduct(
                    data=combined_map_data,
                    input_metadata=self.input_metadata
                )
            case None:
                raise ValueError(f"Could not parse descriptor {self.input_metadata.descriptor}")
            case _:
                raise NotImplementedError(self.input_metadata.descriptor)

        data_product.parent_file_names = sorted(set_of_parent_file_names)

        return [save_data(data_product)]

    def process_full_spin_single_sensor(self, hi_l3_full_spin_dependencies: HiL3SingleSensorFullSpinDependencies,
                                        spice_frame_name: SpiceFrame, combination_strategy: CombinationStrategy) \
            -> RectangularIntensityMapData:
        ram_data_product = process_survival_probabilities(hi_l3_full_spin_dependencies.ram_dependencies,
                                                          spice_frame_name)
        antiram_data_product = process_survival_probabilities(hi_l3_full_spin_dependencies.antiram_dependencies,
                                                              spice_frame_name)

        return combination_strategy.combine_rectangular_intensity_map_data([ram_data_product, antiram_data_product])

    def process_spectral_fit_index(self, hi_l3_spectral_fit_dependencies: HiSpectralIndexDependencies) \
            -> RectangularSpectralIndexMapData:

        return RectangularSpectralIndexMapData(
            spectral_index_map_data=fit_spectral_index_map(hi_l3_spectral_fit_dependencies.map_data.intensity_map_data),
            coords=hi_l3_spectral_fit_dependencies.map_data.coords
        )
