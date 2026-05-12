from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SwapiL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    BAD_FIT = 2**2
    FIT_ERROR = 2**3
    PRELIMINARY_MAG = 2**4
    PUI_FIT_MISSING_UNCERTAINTY = 2**15  # will be removed in PUI update
