import numpy as np
from imap_data_access import upload

from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapDescriptorParts, MapQuantity
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData
from imap_l3_processing.maps.spectral_fit import calculate_spectral_index_for_multiple_ranges
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class LoProcessor(Processor):
    def process(self):
        set_of_parent_file_names = set(self.get_parent_file_names())
        descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                deps = LoL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                spectral_fit_data = perform_spectral_fit(deps.map_data)
                data_product = RectangularSpectralIndexDataProduct(self.input_metadata, spectral_fit_data)
            case None:
                raise ValueError(f"Could not parse descriptor {self.input_metadata.descriptor}")
            case _:
                raise NotImplementedError(self.input_metadata.descriptor)

        data_product.parent_file_names = sorted(set_of_parent_file_names)
        cdf_file = save_data(data_product)
        upload(cdf_file)


def perform_spectral_fit(data: RectangularIntensityMapData) -> RectangularSpectralIndexMapData:
    esa_4_through_7_energy_range = (data.intensity_map_data.energy[3], np.inf)
    return RectangularSpectralIndexMapData(
        spectral_index_map_data=calculate_spectral_index_for_multiple_ranges(data.intensity_map_data,
                                                                             [esa_4_through_7_energy_range]),
        coords=data.coords)
