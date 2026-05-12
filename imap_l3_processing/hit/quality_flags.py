from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class HitL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    PRELIMINARY_MAG = 2 ** 2
