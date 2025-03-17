from spacepy.pycdf import CDF

from imap_l3_processing.cdf.cdf_utils import read_variable
from imap_l3_processing.hi.l3.models import HiL3Data


def read_hi_l2_data(cdf_path) -> HiL3Data:
    with CDF(str(cdf_path)) as cdf:
        return HiL3Data(
            epoch=cdf["Epoch"][...],
            energy=cdf["bin"][...],
            energy_deltas=cdf["bin_boundaries"][...],
            counts=read_variable(cdf["counts"]),
            counts_uncertainty=read_variable(cdf["counts_uncertainty"]),
            epoch_delta=cdf["epoch_delta"][...],
            exposure=read_variable(cdf["exposure"]),
            flux=read_variable(cdf["flux"]),
            lat=cdf["lat"][...],
            lon=cdf["lon"][...],
            sensitivity=read_variable(cdf["sensitivity"]),
            variance=read_variable(cdf["variance"])
        )
