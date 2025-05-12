from imap_data_access import upload

from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies
from imap_l3_processing.map_models import RectangularIntensityMapData, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData, IntensityMapData, SpectralIndexMapData
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class LoProcessor(Processor):
    def process(self):
        deps = LoL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
        spectral_fit_data = spectral_fit_rectangular(deps.map_data)
        data_product = RectangularSpectralIndexDataProduct(self.input_metadata, spectral_fit_data)
        cdf_file = save_data(data_product)
        upload(cdf_file)


def spectral_fit_rectangular(data: RectangularIntensityMapData) -> RectangularSpectralIndexMapData:
    return RectangularSpectralIndexMapData(spectral_index_map_data=spectral_fit(data.intensity_map_data),
                                           coords=data.coords)


def spectral_fit(data: IntensityMapData) -> SpectralIndexMapData:
    raise NotImplementedError("To be imported from elsewhere when complete")
