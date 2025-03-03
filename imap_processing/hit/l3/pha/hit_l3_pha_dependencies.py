from dataclasses import dataclass

from imap_processing.hit.l3.models import HitL1Data
from imap_processing.hit.l3.pha.science.cosine_correction_lookup_table import CosineCorrectionLookupTable
from imap_processing.hit.l3.pha.science.gain_lookup_table import GainLookupTable
from imap_processing.hit.l3.pha.science.hit_event_type_lookup import HitEventTypeLookup
from imap_processing.hit.l3.pha.science.range_fit_lookup import RangeFitLookup
from imap_processing.models import UpstreamDataDependency
from imap_processing.utils import download_dependency

HIT_L1A_EVENT_DESCRIPTOR = "pulse-height-events"
HIT_L3_RANGE_2_COSINE_LOOKUP_DESCRIPTOR = "range-2-cosine-lookup"
HIT_L3_RANGE_3_COSINE_LOOKUP_DESCRIPTOR = "range-3-cosine-lookup"
HIT_L3_RANGE_4_COSINE_LOOKUP_DESCRIPTOR = "range-4-cosine-lookup"
HIT_L3_LO_GAIN_LOOKUP_DESCRIPTOR = "lo-gain-lookup"
HIT_L3_HI_GAIN_LOOKUP_DESCRIPTOR = "hi-gain-lookup"
HIT_L3_RANGE_2_CHARGE_FIT_LOOKUP_DESCRIPTOR = "range-2-charge-fit-lookup"
HIT_L3_RANGE_3_CHARGE_FIT_LOOKUP_DESCRIPTOR = "range_3-charge-fit-lookup"
HIT_L3_RANGE_4_CHARGE_FIT_LOOKUP_DESCRIPTOR = "range-4-charge-fit-lookup"
HIT_L3_EVENT_TYPE_LOOKUP_DESCRIPTOR = "hit-event-type-lookup"


@dataclass
class HitL3PhaDependencies:
    hit_l1_data: HitL1Data
    cosine_correction_lookup: CosineCorrectionLookupTable
    gain_lookup: GainLookupTable
    range_fit_lookup: RangeFitLookup
    event_type_lookup: HitEventTypeLookup

    @classmethod
    def fetch_dependencies(cls, upstream_dependencies: list[UpstreamDataDependency]):
        try:
            l1_events_dependency = next(dependency for dependency in upstream_dependencies)
        except StopIteration:
            raise ValueError(f"Missing {HIT_L1A_EVENT_DESCRIPTOR} dependency.")

        l1_events_cdf_path = download_dependency(l1_events_dependency)
        hit_l1_data = HitL1Data.read_from_cdf(l1_events_cdf_path)

        range_2_cosine_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_2_COSINE_LOOKUP_DESCRIPTOR))
        range_3_cosine_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_3_COSINE_LOOKUP_DESCRIPTOR))
        range_4_cosine_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_4_COSINE_LOOKUP_DESCRIPTOR))
        lo_gain_lookup_path = download_dependency(cls.create_ancillary_dependency(HIT_L3_LO_GAIN_LOOKUP_DESCRIPTOR))
        hi_gain_lookup_path = download_dependency(cls.create_ancillary_dependency(HIT_L3_HI_GAIN_LOOKUP_DESCRIPTOR))
        range_2_charge_fit_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_2_CHARGE_FIT_LOOKUP_DESCRIPTOR))
        range_3_charge_fit_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_3_CHARGE_FIT_LOOKUP_DESCRIPTOR))
        range_4_charge_fit_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_RANGE_4_CHARGE_FIT_LOOKUP_DESCRIPTOR))
        event_type_lookup_path = download_dependency(
            cls.create_ancillary_dependency(HIT_L3_EVENT_TYPE_LOOKUP_DESCRIPTOR))

        cosine_lookup_table = CosineCorrectionLookupTable(range_2_cosine_lookup_path, range_3_cosine_lookup_path,
                                                          range_4_cosine_lookup_path)
        gain_lookup_table = GainLookupTable.from_file(hi_gain_lookup_path, lo_gain_lookup_path)
        range_fit_lookup_table = RangeFitLookup.from_files(range_2_charge_fit_lookup_path,
                                                           range_3_charge_fit_lookup_path,
                                                           range_4_charge_fit_lookup_path)
        event_type_lookup_table = HitEventTypeLookup.from_csv(event_type_lookup_path)
        return cls(hit_l1_data=hit_l1_data,
                   cosine_correction_lookup=cosine_lookup_table,
                   gain_lookup=gain_lookup_table,
                   range_fit_lookup=range_fit_lookup_table,
                   event_type_lookup=event_type_lookup_table)

    @classmethod
    def create_ancillary_dependency(cls, descriptor: str):
        return UpstreamDataDependency(
            descriptor=descriptor,
            instrument="hit",
            data_level="l3",
            start_date=None,
            end_date=None,
            version="latest",
        )
