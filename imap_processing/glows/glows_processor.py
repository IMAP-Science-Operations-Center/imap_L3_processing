import dataclasses

import imap_data_access

from imap_processing.glows.l3a.glows_l3a_dependencies import GlowsL3ADependencies
from imap_processing.glows.l3a.models import GlowsL2Data, GlowsL3LightCurve
from imap_processing.glows.l3a.science.calculate_daily_lightcurve import rebin_lightcurve
from imap_processing.glows.l3a.utils import read_l2_glows_data
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
            data = read_l2_glows_data(l3a_dependencies.data)
            l3a_output = self.process_l3a(data, l3a_dependencies)
            proton_cdf = save_data(l3a_output)
            imap_data_access.upload(proton_cdf)

    def process_l3a(self, data: GlowsL2Data, dependencies: GlowsL3ADependencies) -> GlowsL3LightCurve:
        rebinned_flux, rebinned_exposure = rebin_lightcurve(data.photon_flux, data.histogram_flag_array,
                                                            data.exposure_times, dependencies.number_of_bins)
        return GlowsL3LightCurve(photon_flux=rebinned_flux,
                                 exposure_times=rebinned_exposure,
                                 input_metadata=self.dependencies[0])
