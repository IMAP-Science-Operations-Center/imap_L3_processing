"""@package docstring
Author: Marek Strumik, maro at cbk.waw.pl
Constants (broad-sense understood) are grouped together for convenience in this file
"""

VERSION = "0.2" # version of ground processing software

# ancillary input expected from SDC (based on general IMAP telemetry)
ANC_INPUT_FROM_SDC = {
    # only one file is here now but in the real pipeline there will be probably several files
    'anc_data_file_name': 'data_ancillary/imap_l1_anc_sc_Merged_2010-2030_mockByGlowsTeam001.csv'
}

# ancillary input expected from IMAP/GLOWS Instrument Team
ANC_INPUT_FROM_INSTRUMENT_TEAM = {
    'conversion_table_for_anc_data':
        'data_ancillary/imap_glows_conversion_table_for_anc_data_v001.json',
    'calibration_data': 'data_ancillary/imap_glows_calibration_data_v001.json',
    'settings': 'data_ancillary/imap_glows_pipeline_settings_v001.json',
    'map_of_uv_sources': 'data_ancillary/imap_glows_map_of_uv_sources_v001.dat',
    'map_of_excluded_regions': 'data_ancillary/imap_glows_map_of_excluded_regions_v001.dat',
    'exclusions_by_instr_team': 'data_ancillary/imap_glows_exclusions_by_instr_team_v001.dat',
    'suspected_transients': 'data_ancillary/imap_glows_suspected_transients_v001.dat'
}

# subsecond limit for GLOWS clock (and consequently also onboard-interpolated IMAP clock)
SUBSECOND_LIMIT = 2_000_000

# angular radius of IMAP/GLOWS scanning circle [deg]
SCAN_CIRCLE_ANGULAR_RADIUS = 75.0

# definition of bit fields for histogram flag array at L1b
# numbers indicate bitwise left_shift for a given field
L1B_L2_HIST_FLAG_FIELDS = {
    # bad-angle bits (related to individual bins)
    'is_close_to_uv_source': 0,
    'is_inside_excluded_region': 1,
    'is_excluded_by_instr_team': 2,
    'is_suspected_transient': 3
}

# instrument emulator sometimes does not set the imap_start_time properly, then we have to use
# glows_start_time, MAIN_TIME_FIELD was defined to switch quickly between imap_start_time and
# glows_start_time for some places in the code where correct values of imap_start_time are crucial
# TMP_SOLUTION, NEEDS TO BE CHANGED WHEN imap_start_time will be correct
MAIN_TIME_FIELD = 'imap_start_time'

# number of bins for helioglow lightcurve at L3
L3_N_BINS = 90
