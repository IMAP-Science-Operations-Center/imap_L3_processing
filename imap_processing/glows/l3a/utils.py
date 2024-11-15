from spacepy.pycdf import CDF

from imap_processing.glows.l3a.models import GlowsL2Data


def read_l2_glows_data(cdf: CDF) -> GlowsL2Data:
    return GlowsL2Data(photon_flux=cdf['photon_flux'][...],
                       start_time=cdf['start_time'][...],
                       end_time=cdf['end_time'][...],
                       histogram_flag_array=cdf['histogram_flag_array'][...].astype(bool),
                       flux_uncertainties=cdf['flux_uncertainties'][...],
                       spin_angle=cdf['spin_angle'][...],
                       exposure_times=cdf['exposure_times'][...])
