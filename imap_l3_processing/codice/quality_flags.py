from imap_processing.quality_flags import CommonFlags, FlagNameMixin


class CodiceL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    PRELIMINARY_MAG = 2**2
