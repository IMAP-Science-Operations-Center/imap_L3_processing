from imap_processing.quality_flags import CommonFlags, FlagNameMixin


class CodiceL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    GAIN_TEST = 2**0
    MCP_CURRENT = 2**1
    PRELIMINARY_MAG = 2**2


CodiceL2FlagsMask = CodiceL3Flags.GAIN_TEST | CodiceL3Flags.MCP_CURRENT
