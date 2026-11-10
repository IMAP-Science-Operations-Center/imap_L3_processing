from imap_processing.quality_flags import CommonFlags, FlagNameMixin


class MapL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    PREDICTIVE_EPHEMERIS = 2**15
    NOMINAL_ALPHA_PROTON_RATIO = 2**2
    PERSISTED_LAST_POINT = 2**3
