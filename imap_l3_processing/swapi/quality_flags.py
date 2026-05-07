from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SwapiL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    EPHEMERIS_GAP = 2**2
    HI_CHI_SQ = 2**3
    PUI_FIT_MISSING_UNCERTAINTY = 2**4
    STALE_PROTON = 2**5
    PRELIMINARY_MAG = 2**6
    MAG_GAP = 2**7
    FIT_FAILED = 2**8
