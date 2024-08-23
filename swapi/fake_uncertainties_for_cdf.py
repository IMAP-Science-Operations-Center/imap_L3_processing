import numpy as np
from spacepy import pycdf

def fake_uncertainties(cdf_path):
    original_cdf = pycdf.CDF(cdf_path, readonly=False)

    coincidence_count_rates: np.ndarray = original_cdf["swp_coin_rate"][...]
    uncertainties = np.sqrt(6 * coincidence_count_rates)
    original_cdf["swp_coin_unc"] = uncertainties

    original_cdf.save()