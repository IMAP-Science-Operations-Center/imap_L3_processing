import numba

SWAPI_DEADTIME_S = 183.7e-9


@numba.njit
def deadtime_factor(true_rate):
    return 1.0 / (1.0 + SWAPI_DEADTIME_S * true_rate)
