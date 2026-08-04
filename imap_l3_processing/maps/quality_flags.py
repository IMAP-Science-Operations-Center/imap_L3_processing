from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class MapL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    PREDICTIVE_EPHEMERIS = 2**15
