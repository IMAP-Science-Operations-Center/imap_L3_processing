# todo placeholder needed function until we are ready to fully integrate L3B code

PHISICAL_CONSTANTS = {
    'jd_carrington_first': 2.398167329 * 10 ** 6,
    'carrington_time': 27.2753
}


def carrington(jd):
    '''
    Converts time in Julian Day format into Carrington Rotation number
    '''
    ans = (jd - PHISICAL_CONSTANTS['jd_carrington_first']) / PHISICAL_CONSTANTS['carrington_time'] + 1
    return ans


def jd_fm_Carrington(carr):
    '''
    calculates time in format of Julian Days for a given Carrington Rotation number
    '''
    ans = (carr - 1) * PHISICAL_CONSTANTS['carrington_time'] + PHISICAL_CONSTANTS['jd_carrington_first']
    return ans
