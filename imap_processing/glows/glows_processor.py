import dataclasses

import imap_data_access
import numpy as np
from uncertainties import unumpy

from imap_processing.constants import ONE_SECOND_IN_NANOSECONDS
from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.glows.l3a.models import GlowsL3LightCurve
from imap_processing.glows.l3a.science.calculate_daily_lightcurve import rebin_lightcurve
from imap_processing.processor import Processor
from imap_processing.utils import save_data


class GlowsProcessor(Processor):

    def process(self):
        dependencies = [
            dataclasses.replace(dep, start_date=self.input_metadata.start_date, end_date=self.input_metadata.end_date)
            for dep in
            self.dependencies]

        if self.input_metadata.data_level == "l3a":
            l3a_dependencies = GlowsL3ADependencies.fetch_dependencies(dependencies)
            l3a_output = self.process_l3a(l3a_dependencies)
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)

    def process_l3a(self, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        data = dependencies.data
        flux_with_uncertainty = unumpy.uarray(data.photon_flux, data.flux_uncertainties)
        rebinned_flux, rebinned_exposure = rebin_lightcurve(dependencies.time_independent_background_table,
                                                            flux_with_uncertainty, data.ecliptic_lat, data.ecliptic_lon,
                                                            data.histogram_flag_array,
                                                            data.exposure_times, dependencies.number_of_bins,
                                                            dependencies.time_dependent_background)

        epoch = data.start_time + (data.end_time - data.start_time) / 2
        epoch_delta = (data.end_time - data.start_time).total_seconds() * ONE_SECOND_IN_NANOSECONDS / 2
        return GlowsL3LightCurve(
            photon_flux=rebinned_flux.reshape(1, -1),
            exposure_times=rebinned_exposure.reshape(1, -1),
            input_metadata=self.input_metadata.to_upstream_data_dependency(self.dependencies[0].descriptor),
            epoch=np.array([epoch]),
            epoch_delta=np.array([epoch_delta])
        )
