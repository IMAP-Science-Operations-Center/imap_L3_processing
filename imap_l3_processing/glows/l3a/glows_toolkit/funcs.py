"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
Various functions that are not methods to classes
"""
import datetime
import numpy as np
from .imap_time import SpacecraftTime
from .constants import SUBSECOND_LIMIT

def calibration_factor(calibration_file_name, date_string):
    """
    Compute calibration factor cps_per_R to convert cps to the absolute physical units of Rayleigh
    THERE IS A STRONG ASSUMPION THROUGHOUT THE ENTIRE CODE THAT THE CALIBRATION IS CONSTANT FOR A
    GIVEN OBSERVATIONAL DAY. IT IS APPLIED TO L2 and L3A, WHICH ARE ACCUMULATED OVER THE ENTIRE
    OBSERVATIONAL DAYS.
    Args:
        calibration_file_name: name of ancillary file with calibration data provided by the GLOWS team
        date_string: string representing date in the following format 'yyyy-mm-ddThh:mm:ss'
    """
    date_array = np.loadtxt(calibration_file_name, usecols=[0], dtype=np.datetime64)
    cps_per_R_array = np.loadtxt(calibration_file_name, usecols=[1])
    idx = np.nonzero(date_array <= np.datetime64(date_string))[0][-1]
    return cps_per_R_array[idx]

def check_if_contains_actual_data(file_name):
    """
    Check if there is any actual data in a file apart from header
    Non-empty lines NOT starting with "#" are considered as actual-data file
    Args:
        file_name: name of file to be checked
    """
    f = open(file_name, 'r')
    line = f.readline()
    is_data_in_file = False
    while line:
        if line.lstrip()[0] != '#' and len(line.lstrip()) > 0:
            is_data_in_file = True
            break
        line = f.readline()
    f.close()
    return is_data_in_file

def decode_ancillary_parameter(name, decoding_table, encoded_value):
    """
    Decode parameters collected onboard, which are encoded as int at
        L0 and L1a, but should be decoded to physical units at L1b
    Args:
        name: name of parameter, e.g., 'filter_temperature'
        decoding_table: dictionary created from JSON file provided by GLOWS Instrument Team
        l1a_data: L1a data provided as dictionary
    """
    params = decoding_table[name] # extract decoding params for a given parameter name

    # compute parameters A and B
    p_a = (2**params['n_bits'] - 1)/(params['max'] - params['min'])
    p_b = -params['min']*p_a

    # decode int-encoded value
    decoded_value = (encoded_value-p_b)/p_a

    return decoded_value

def decode_ancillary_parameters_avg_stddev(name, decoding_table, l1a_data):
    """
    Decode parameters collected onboard (average and variance), which are encoded as int at
        L0 and L1a, but should be decoded to physical units at L1b
    Additionally sqrt of variance is computed to get std deviation
    Args:
        name: name of parameter, e.g., 'filter_temperature'
        decoding_table: dictionary created from JSON file provided by GLOWS Instrument Team
        l1a_data: L1a data provided as dictionary
    """
    params = decoding_table[name] # extract decoding params for a given parameter name

    # compute parameters A and B from Sec. "On-board encoding and ground decoding of measured quantities"
    # in the algorithm document
    p_a = (2**params['n_bits'] - 1)/(params['max'] - params['min'])
    p_b = -params['min']*p_a

    # decode int values for the average and variance
    average = (l1a_data[name + '_average']-p_b)/p_a
    variance = l1a_data[name + '_variance']/(p_a*p_a)

    # compute standard deviation
    std_deviation = np.sqrt(variance)

    return average, std_deviation

def flags_deserialize(raw):
    """
    Convert raw-int-number-encoded GLOWS flags  to a dictionary
    This procedure is for histogram flags generated onboard
    Args:
        raw: flags encoded as bit fields of an integer
    """
    assert raw < 65536, 'Argument of flags_deserialize() is wrong'
    flags = {}
    flags['is_pps_missing'] = bool((raw >> 0) & 1)
    flags['is_time_status_missing'] = bool((raw >> 1) & 1)
    flags['is_phase_missing'] = bool((raw >> 2) & 1)
    flags['is_spin_period_missing'] = bool((raw >> 3) & 1)
    flags['is_overexposed'] = bool((raw >> 4) & 1)
    flags['is_direct_event_non_monotonic'] = bool((raw >> 5) & 1)
    flags['is_night'] = bool((raw >> 6) & 1)
    flags['is_hv_test_in_progress'] = bool((raw >> 7) & 1)
    flags['is_test_pulse_in_progress'] = bool((raw >> 8) & 1)
    flags['is_memory_error_detected'] = bool((raw >> 9) & 1)
    return flags

def read_l0_data(pkts_file_name):
    """
    Read L0 data as an input data for L1a
    Args:
        pkts_file_name: name of of the file with CCSDS packets
    """
    # open binary pkts file containing CCSDS packets with histograms
    file_handler = open('%s' % pkts_file_name, mode='rb')
    # read binary file with histograms (CCSDS headers are assumed to be included)
    histogram_telemetry_packets = file_handler.read()
    file_handler.close()
    return histogram_telemetry_packets # byte array

def time_sec_subsec_to_utc(seconds: int, subseconds: int) -> datetime:
    """
    Convert time (seconds-and-subseconds-from-2010-01-01 format) to datatime object
    Args:
        seconds:
        subseconds:
    """
    result = SpacecraftTime.spacecraft_epoch_to_date(seconds)
    result = result + datetime.timedelta(microseconds=subseconds/2)
    return result # datatime object

def time_sec_subsec_to_float64(seconds: int, subseconds: int):
    """
    Convert time (seconds-and-subseconds-from-2010-01-01 format) to
    a double-precision floating-point number, where the decimal-fraction
    part corresponds to fraction of second
    Args:
        seconds:
        subseconds:
    """
    # python default floating-point type is 64-bit double precision
    result = seconds + subseconds/SUBSECOND_LIMIT
    return result
