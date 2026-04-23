from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SweL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    FALLBACK_POTENTIAL_ESTIMATE = 2 ** 2
    BACKUP_SPLINE_UNRESOLVED = 2 ** 3
    POTENTIAL_FIT_UNCONVERGED = 2 ** 4
    BREAKPOINT_FIT_UNCONVERGED = 2 ** 5
    ULTRA_HV_OFF = 2 ** 6
    TEMPERATURE_OUTLIER = 2 ** 7
    PRELIMINARY_MAG = 2 ** 8
    FALLBACK_SWAPI_SPEED = 2 ** 9
    NEGATIVE_MOMENT = 2 ** 10
    PREDICTIVE_EPHEMERIS = 2 ** 15

