from spacepy.pycdf import CDF

from imap_processing.hit.l3.models import HitL2Data


def read_l2_hit_data(cdf: CDF) -> HitL2Data:
    return HitL2Data(
        epoch=cdf.raw_var("Epoch")[...],
        epoch_delta=cdf["Epoch_DELTA"][...],
        flux=cdf["R26A_H_SECT_Flux"][...],
        count_rates=cdf["R26A_H_SECT_Rate"][...],
        uncertainty=cdf["R26A_H_SECT_Uncertainty"][...],
    )

