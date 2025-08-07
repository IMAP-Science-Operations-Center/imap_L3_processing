from pathlib import Path

from imap_l3_processing.hi.l3.hi_l3_combined_sensor_dependencies import HiL3CombinedMapDependencies
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralIndexDependencies
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies, \
    HiL3SingleSensorFullSpinDependencies
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapQuantity, MapDescriptorParts, \
    SurvivalCorrection, \
    SpinPhase, Sensor, Duration
from imap_l3_processing.maps.map_models import RectangularSpectralIndexDataProduct, RectangularSpectralIndexMapData, \
    RectangularIntensityMapData, RectangularIntensityDataProduct, \
    combine_rectangular_intensity_map_data
from imap_l3_processing.maps.spectral_fit import fit_spectral_index_map
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from imap_l3_processing.models import Instrument
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class HiProcessor(Processor):
    def process(self) -> list[Path]:
        set_of_parent_file_names = set(self.get_parent_file_names())

        parsed_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match parsed_descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                hi_l3_spectral_fit_dependencies = HiL3SpectralIndexDependencies.fetch_dependencies(self.dependencies)
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
                    data=process_survival_probabilities(hi_l3_survival_probabilities_dependencies),
                    input_metadata=self.input_metadata,
                )
                set_of_parent_file_names.update(
                    p.name for p in hi_l3_survival_probabilities_dependencies.dependency_file_paths)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    sensor=Sensor.Hi90 | Sensor.Hi45,
                                    spin_phase=SpinPhase.FullSpin):
                hi_l3_full_spin_dependencies = HiL3SingleSensorFullSpinDependencies.fetch_dependencies(
                    self.dependencies)
                combined_map = self.process_full_spin_single_sensor(hi_l3_full_spin_dependencies)
                data_product = RectangularIntensityDataProduct(
                    data=combined_map,
                    input_metadata=self.input_metadata
                )
                set_of_parent_file_names.update(p.name for p in hi_l3_full_spin_dependencies.dependency_file_paths)
            case MapDescriptorParts(sensor=Sensor.HiCombined,
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

        return [save_data(data_product)]

    def process_full_spin_single_sensor(self, hi_l3_full_spin_dependencies: HiL3SingleSensorFullSpinDependencies) \
            -> RectangularIntensityMapData:
        ram_data_product = process_survival_probabilities(hi_l3_full_spin_dependencies.ram_dependencies)
        antiram_data_product = process_survival_probabilities(hi_l3_full_spin_dependencies.antiram_dependencies)

        return combine_rectangular_intensity_map_data([ram_data_product, antiram_data_product])

    def process_spectral_fit_index(self, hi_l3_spectral_fit_dependencies: HiL3SpectralIndexDependencies) \
            -> RectangularSpectralIndexMapData:

        return RectangularSpectralIndexMapData(
            spectral_index_map_data=fit_spectral_index_map(hi_l3_spectral_fit_dependencies.map_data.intensity_map_data),
            coords=hi_l3_spectral_fit_dependencies.map_data.coords
        )
