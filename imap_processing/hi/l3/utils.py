from spacepy.pycdf import CDF

from imap_processing.hi.l3.models import HiL3Data


def read_hi_l3_data(cdf_path) -> HiL3Data:
    with CDF(str(cdf_path)) as cdf:
        return HiL3Data(
            epoch=cdf["Epoch"][...],
            energy=cdf["bin"][...],
            energy_deltas=cdf["bin_boundaries"][...],
            counts=cdf["counts"][...],
            counts_uncertainty=cdf["counts_uncertainty"][...],
            epoch_delta=cdf["epoch_delta"][...],
            exposure=cdf["exposure"][...],
            flux=cdf["flux"][...],
            lat=cdf["lat"][...],
            lon=cdf["lon"][...],
            sensitivity=cdf["sensitivity"][...],
            variance=cdf["variance"][...]
        )
