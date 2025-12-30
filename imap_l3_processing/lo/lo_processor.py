import dataclasses
from pathlib import Path

import numpy as np
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapDescriptorParts, MapQuantity, \
    SurvivalCorrection, ReferenceFrame
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData, RectangularIntensityDataProduct, InputRectangularPointingSet
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.spectral_fit import fit_spectral_index_map
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from imap_l3_processing.models import Instrument
from imap_l3_processing.utils import save_data


class LoProcessor(MapProcessor):
    def process(self, spice_frame_name: SpiceFrame = SpiceFrame.ECLIPJ2000) -> list[Path]:
        set_of_parent_file_names = set(self.get_parent_file_names())
        descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                deps = LoL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                spectral_fit_data = perform_spectral_fit(deps.map_data)
                data_product = RectangularSpectralIndexDataProduct(self.input_metadata, spectral_fit_data)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    reference_frame=ReferenceFrame.Spacecraft | ReferenceFrame.Heliospheric):
                deps = HiLoL3SurvivalDependencies.fetch_dependencies(self.dependencies, Instrument.IMAP_LO)
                deps.l1c_data = list(map(self._collapse_pset_dimension, deps.l1c_data))
                data = process_survival_probabilities(deps, spice_frame_name)

                data_product = RectangularIntensityDataProduct(self.input_metadata, data)
                set_of_parent_file_names.update(path.name for path in deps.dependency_file_paths)
            case None:
                raise ValueError(f"Could not parse descriptor {self.input_metadata.descriptor}")
            case _:
                raise NotImplementedError(self.input_metadata.descriptor)

        data_product.parent_file_names = sorted(set_of_parent_file_names)
        return [save_data(data_product)]

    @staticmethod
    def _collapse_pset_dimension(pset: InputRectangularPointingSet) -> InputRectangularPointingSet:
        return dataclasses.replace(pset, exposure_times=np.sum(pset.exposure_times, axis=3))

def perform_spectral_fit(data: RectangularIntensityMapData) -> RectangularSpectralIndexMapData:
    return RectangularSpectralIndexMapData(
        spectral_index_map_data=fit_spectral_index_map(data.intensity_map_data),
        coords=data.coords
    )
