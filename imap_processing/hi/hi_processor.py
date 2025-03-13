import numpy as np

from imap_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies, HI_L3_DESCRIPTOR
from imap_processing.hi.l3.models import HiL3SpectralIndexDataProduct
from imap_processing.hi.l3.science.spectral_fit import spectral_fit
from imap_processing.processor import Processor
from imap_processing.utils import save_data


class HiProcessor(Processor):
    def process(self):
        hi_l3_dependencies = HiL3Dependencies.fetch_dependencies(self.dependencies)
        data_product = self._process_spectral_fit_index(hi_l3_dependencies)
        save_data(data_product)

    def _process_spectral_fit_index(self, hi_l3_dependencies):
        hi_l3_data = hi_l3_dependencies.hi_l3_data

        epochs = hi_l3_data.epoch
        energy = hi_l3_data.energy
        fluxes = hi_l3_data.flux
        lons = hi_l3_data.lon
        lats = hi_l3_data.lat
        variances = hi_l3_data.variance

        gammas = spectral_fit(len(epochs), len(lons), len(lats), fluxes, variances, energy)

        data_product = HiL3SpectralIndexDataProduct(self.input_metadata.to_upstream_data_dependency(HI_L3_DESCRIPTOR),
                                                    hi_l3_dependencies.hi_l3_data.epoch,
                                                    hi_l3_dependencies.hi_l3_data.energy,
                                                    hi_l3_dependencies.hi_l3_data.variance,
                                                    hi_l3_dependencies.hi_l3_data.flux,
                                                    hi_l3_dependencies.hi_l3_data.lat,
                                                    hi_l3_dependencies.hi_l3_data.lon,
                                                    gammas,
                                                    hi_l3_dependencies.hi_l3_data.energy_deltas,
                                                    hi_l3_dependencies.hi_l3_data.counts,
                                                    hi_l3_dependencies.hi_l3_data.counts_uncertainty,
                                                    hi_l3_dependencies.hi_l3_data.epoch_delta,
                                                    hi_l3_dependencies.hi_l3_data.exposure,
                                                    hi_l3_dependencies.hi_l3_data.sensitivity)

        return data_product
