import numpy as np

from imap_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies, HI_L3_DESCRIPTOR
from imap_processing.hi.l3.models import HiL3SpectralIndexDataProduct
from imap_processing.hi.l3.science.mpfit import mpfit
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

        initial_parameters = (1, 1)

        gammas = np.full((len(epochs), len(lons), len(lats)), fill_value=np.nan, dtype=float)
        for epoch in range(len(epochs)):
            for lon in range(len(lons)):
                for lat in range(len(lats)):
                    flux = fluxes[epoch][lon][lat]
                    variance = variances[epoch][lon][lat]
                    keywords = {'xval': energy, 'yval': flux, 'errval': variance}
                    _, gamma = mpfit(self._fit_function, initial_parameters, keywords, nprint=0).params
                    gammas[epoch][lon][lat] = gamma

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

    def _fit_function(self, params, **kwargs):
        A, B = params
        x = kwargs['xval']
        y = kwargs['yval']
        err = kwargs['errval']

        model = A * np.power(x, -B)

        status = 0
        residuals = (y - model) / err

        return status, residuals
