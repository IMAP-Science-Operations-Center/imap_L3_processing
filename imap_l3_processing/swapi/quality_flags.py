from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SwapiL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    SWP_SW_ANGLES_ESTIMATED = 2 ** 2
    HI_CHI_SQ = 2 ** 3
