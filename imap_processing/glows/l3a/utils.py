from spacepy.pycdf import CDF

from imap_processing.glows.l3a.models import GlowsL2Data


def read_l2_glows_data(cdf: CDF) -> GlowsL2Data:
    assert 1 == cdf['photon_flux'].shape[0], "Level 2 file should have only one histogram"

    return GlowsL2Data(photon_flux=cdf['photon_flux'][0],
                       start_time=cdf['start_time'][0],
                       end_time=cdf['end_time'][0],
                       histogram_flag_array=cdf['histogram_flag_array'][0].astype(bool),
                       flux_uncertainties=cdf['flux_uncertainties'][0],
                       spin_angle=cdf['spin_angle'][0],
                       exposure_times=cdf['exposure_times'][0],
                       epoch=cdf['epoch'][0],
                       ecliptic_lat=cdf['ecliptic_lat'][0],
                       ecliptic_lon=cdf['ecliptic_lon'][0],
                       )
