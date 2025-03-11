import numpy as np

from imap_processing.hi.l3.hi_l3_dependencies import HiL3Dependencies
from imap_processing.hi.l3.science.mpfit import mpfit
from imap_processing.processor import Processor


class HiProcessor(Processor):
    def process(self):
        hi_l3_dependencies = HiL3Dependencies.fetch_dependencies(self.dependencies)
        self._process_spectral_fit_index(hi_l3_dependencies)

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
                    _, gamma = mpfit(self._fit_function, initial_parameters, keywords).params
                    gammas[epoch][lon][lat] = gamma
        return gammas

    def _fit_function(self, params, **kwargs):
        A, B = params
        x = kwargs['xval']
        y = kwargs['yval']
        err = kwargs['errval']

        model = A * np.power(x, -B)

        status = 0
        residuals = (y - model) / err

        return status, residuals
